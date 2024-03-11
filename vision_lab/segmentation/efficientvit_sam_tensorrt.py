# https://github.com/xuanlinli17/efficientvit
# https://github.com/xuanlinli17/efficientvit/blob/master/applications/sam.md
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch2trt import TRTModule
from torchvision.transforms.functional import resize
from real_robot.utils.logger import get_logger

from ..utils.io import load_image_arrays
from ..utils.timer import timer


class EfficientViT_SAM_TensorRT:
    """SAM for object segmentation"""

    CKPT_DIR = Path(os.getenv("EFFICIENTVIT_SAM_TENSORRT_CKPT_DIR", Path.home() / "checkpoints/EfficientViT_SAM_TensorRT"))

    CKPT_TYPES = ["l0", "l1", "l2", "xl0", "xl1"]

    logger = get_logger("EfficientViT_SAM")

    def __init__(
        self,
        model_variant="xl0",
        prompt_mode="points",
        max_point_box_batch_num=8,
        max_points_per_prompt=1,
        device="cuda",
    ):
        self.logger.info('Using EfficientViT-SAM-TensorRT model variant: "%s"; prompt mode: "%s"', model_variant, prompt_mode)
        assert model_variant in self.CKPT_TYPES, f"{model_variant = }"
        assert prompt_mode in ["points", "boxes"], f"{prompt_mode = }"
        self.model_variant = model_variant
        self.prompt_mode = prompt_mode
        self.max_point_box_batch_num = max_point_box_batch_num
        self.max_points_per_prompt = max_points_per_prompt

        # setup cuda context
        assert "cuda" in device, "cpu inference is not supported"
        self.device = device
        # the model uses the first visible cuda device by default, so we don't specify in the args
        self.load_model()

    @timer
    def load_model(self):
        self.encoder_engine = self.CKPT_DIR / f"{self.model_variant}_encoder.engine"
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(self.encoder_engine, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.encoder_engine = TRTModule(
            engine,
            input_names=["input_image"],
            output_names=["image_embeddings"]
        )

        self.decoder_engine = self.CKPT_DIR / f"{self.model_variant}_decoder.engine"
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(self.decoder_engine, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.decoder_engine = TRTModule(
            engine,
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=["masks", "iou_predictions"]
        )

    @timer
    @torch.no_grad()
    def __call__(
        self,
        images: str | list[str] | np.ndarray | list[np.ndarray],
        boxes: Optional[
            torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray]
        ] = None,
        *,
        points: Optional[
            torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray]
        ] = None,
        point_labels: Optional[
            int | torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray]
        ] = None,
        return_on_cpu=False,
        verbose=False,
    ) -> (
        tuple[torch.Tensor, np.float32 | np.ndarray]
        | tuple[list[torch.Tensor], list[np.ndarray]]
    ):
        """
        :param images: Input RGB images, can be
                       a string path, a list of string paths,
                       a [H, W, 3] np.uint8 np.ndarray,
                       a list of [H, W, 3] np.uint8 np.ndarray,
                       a [n_images, H, W, 3] np.uint8 np.ndarray
        :param boxes: (n_images) list of pred_bbox as XYXY pixel coordinates
                      [B, 4] or [4,] int/float32 np.ndarray/torch.Tensor
        :param points: (n_images) list of point prompts as XY pixel coordinates
                       [B, n_points, 2] or [n_points, 2] or [2,]
                       int/float32 np.ndarray/torch.Tensor
        :param point_labels: (n_images) list of point prompt labels.
                             1 is foreground, 0 is background, -1 is ignored.
                             [B, n_points] or [n_points,] int np.ndarray/torch.Tensor
        :param return_on_cpu: whether to return masks as cuda Tensor or numpy array
        :param verbose: whether to print debug info
        :return masks: (n_images) list of predicted mask
                       [B, H, W] or [H, W] torch.bool cuda Tensor
        :return pred_ious: (n_images) list of [B,] or () np.float32 np.ndarray

        Here B is the number of masks per image.
        """
        masks, pred_ious = [], []
        assert boxes is not None or points is not None, "Need boxes or points prompt"

        # Process images and boxes
        images, multiple_images = load_image_arrays(images)
        if boxes is not None:
            # list[array[4,]], list[array[B, 4]]
            # array[4,], array[B, 4], array[n_images, B, 4]
            assert self.prompt_mode == "boxes"
            if squeeze_return := not isinstance(boxes, list):
                squeeze_return = boxes.ndim == 1  # boxes has shape [4,]
                if boxes.ndim < 3:
                    boxes = [boxes]  # type: ignore
            assert len(images) == len(boxes), f"{len(images) = } {len(boxes) = }"  # type: ignore
        if points is not None:
            assert self.prompt_mode == "points"
            assert point_labels is not None, "Need point_labels for point prompts"
            # list[array[2,]], list[array[N, 2]], list[array[B, N, 2]]
            # array[2,], array[N, 2], array[B, N, 2], array[n_images, B, N, 2]
            if squeeze_return := not isinstance(points, list):
                squeeze_return = points.ndim in [1, 2]  # [2,] or [N, 2]
                if points.ndim < 4:
                    points, point_labels = [points], [point_labels]  # type: ignore
            assert (
                len(images) == len(points) == len(point_labels)  # type: ignore
            ), f"{len(images) = } {len(points) = } {len(point_labels) = }"  # type: ignore

        # run EfficientViT-SAM model on 1-image batch
        for i, image in enumerate(images):
            origin_image_size = image.shape[:2]
            if self.model_variant in ["l0", "l1", "l2"]:
                image = preprocess(image, img_size=512, device=self.device)
            elif self.model_variant in ["xl0", "xl1"]:
                image = preprocess(image, img_size=1024, device=self.device)
            else:
                raise NotImplementedError
            image_embedding = self.encoder_engine(image)
            image_embedding = image_embedding[0].reshape(1, 256, 64, 64)

            input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)

            if points is not None:
                point, point_label = points[i], point_labels[i]  # type: ignore
                if point.ndim == 1:
                    point = point[None, None, :]  # [1, 1, 2]
                    point_label = np.asarray(point_label).reshape(1, -1)  # [1, 1]
                elif point.ndim == 2:
                    point = point[None, ...]  # [1, N, 2]
                    point_label = np.asarray(point_label).reshape(1, -1)  # [1, N]
                
                assert len(point) == len(point_label), (
                    f"Mismatch {len(point) = } {len(point_label) = }"
                )
                assert len(point) <= self.max_point_box_batch_num, (
                    f"Number of prompts {len(point) = } is more than the predetermined batch size {self.max_point_box_batch_num = }"
                )
                assert point.shape[1] <= self.max_points_per_prompt, (
                    f"Number of points per prompt {point.shape[1] = } is more than the predetermined value {self.max_points_per_prompt = }"
                )
                point = apply_coords(point, origin_image_size, input_size).astype(np.float32)

            elif boxes is not None:
                box = boxes[i]
                if box.ndim == 1:
                    box = box[None, :]  # [1, 4]
                box = apply_boxes(box, origin_image_size, input_size).astype(np.float32)
                box_label = np.array([[2, 3] for _ in range(box.shape[0])], dtype=np.float32).reshape((-1, 2))
                point = box
                point_label = box_label
                assert len(point) <= self.max_point_box_batch_num, (
                    f"Number of prompts {len(point) = } is more than the predetermined batch size {self.max_point_box_batch_num = }"
                )

            inputs = (image_embedding, torch.from_numpy(point).to(self.device), torch.from_numpy(point_label).to(self.device))
            low_res_mask, pred_iou = self.decoder_engine(*inputs)
            low_res_mask = low_res_mask.reshape(1, -1, 256, 256)

            mask = mask_postprocessing(low_res_mask, origin_image_size)[0] # [B, H, W]
            mask = mask > 0.0
            mask = mask.cpu().numpy() if return_on_cpu else mask
            pred_iou = pred_iou.reshape(-1) # [B]

            if not multiple_images:  # single input image
                return (
                    (mask[0], pred_iou[0]) if squeeze_return else (mask, pred_iou)
                )

            masks.append(mask)
            pred_ious.append(pred_iou)

        return masks, pred_ious
    
    def __del__(self):
        self.cuda_driver_context.pop()



# utils
    
class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


def preprocess(x, img_size, device):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x).to(device)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0)

    return x


def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    img_size = 1024
    masks = torch.tensor(masks)
    orig_im_size = torch.tensor(orig_im_size)

    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_coords(coords, original_size, new_size):
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(boxes, original_size, new_size):
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
    return boxes

