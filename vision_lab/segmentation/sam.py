# python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gdown
import numpy as np
import torch
from real_robot.utils.logger import get_logger
from segment_anything import sam_model_registry  # type: ignore
from segment_anything.utils.transforms import ResizeLongestSide  # type: ignore

from ..utils.io import load_image_arrays
from ..utils.timer import timer


class SAM:
    """SAM for object segmentation"""

    CKPT_DIR = Path(os.getenv("SAM_CKPT_DIR", Path.home() / "checkpoints/SAM"))

    CKPT_PATHS = {
        "vit_b": CKPT_DIR / "sam_vit_b_01ec64.pth",
        "vit_l": CKPT_DIR / "sam_vit_l_0b3195.pth",
        "vit_h": CKPT_DIR / "sam_vit_h_4b8939.pth",
    }

    CKPT_GDOWN_PARAMS = {
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "hash": "md5:01ec64d29a2fca3f0661936605ae66f8",
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "hash": "md5:0b3195507c641ddb6910d2bb5adee89c",
        },
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "hash": "md5:4b8939a88964f0f4ff5f5b2642c598a6",
        },
    }

    logger = get_logger("SAM")

    def __init__(
        self,
        model_variant="vit_h",
        device="cuda",
    ):
        self.logger.info('Using SAM model variant: "%s"', model_variant)

        self.ckpt_path = self.CKPT_PATHS[model_variant]
        self.model_variant = model_variant

        self.device = device

        self.load_model()

    @timer
    def load_model(self):
        # Download checkpoint if not found
        gdown.cached_download(
            path=self.ckpt_path,
            **self.CKPT_GDOWN_PARAMS[self.model_variant],  # type: ignore
        )

        self.model = sam_model_registry[self.model_variant](checkpoint=self.ckpt_path)
        self.resize_transform = ResizeLongestSide(self.model.image_encoder.img_size)

        self.model = self.model.to(self.device)
        self.model.eval()

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
                             1 is foreground, 0 is background.
                             [B, n_points] or [n_points,] int np.ndarray/torch.Tensor
        :param return_on_cpu: whether to return masks as cuda Tensor or numpy array
        :param verbose: whether to print debug info
        :return masks: (n_images) list of predicted mask
                       [B, H, W] or [H, W] torch.bool cuda Tensor
        :return pred_ious: (n_images) list of [B,] or () np.float32 np.ndarray
        """
        masks, pred_ious = [], []
        assert boxes is not None or points is not None, "Need boxes or points prompt"

        with torch.cuda.device(self.device):
            # Process images and boxes
            images, multiple_images = load_image_arrays(images)
            if boxes is not None:
                # list[array[4,]], list[array[B, 4]]
                # array[4,], array[B, 4], array[n_images, B, 4]
                if squeeze_return := not isinstance(boxes, list):
                    squeeze_return = boxes.ndim == 1  # boxes has shape [4,]
                    if boxes.ndim < 3:
                        boxes = [boxes]  # type: ignore
                assert len(images) == len(boxes), f"{len(images) = } {len(boxes) = }"  # type: ignore
            if points is not None:
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

            # run SAM model on 1-image batch (on 11GB GPU)
            for i, image in enumerate(images):
                image_shape = image.shape[:2]

                processed_image = self.resize_transform.apply_image(image)
                processed_image = (
                    torch.as_tensor(processed_image, device=self.device)
                    .permute(2, 0, 1)
                    .contiguous()
                )

                input_dict = {"image": processed_image, "original_size": image_shape}
                batch_size = -1
                if boxes is not None:
                    input_dict["boxes"] = boxes_torch = (
                        self.resize_transform.apply_boxes_torch(
                            torch.as_tensor(boxes[i], device=self.device), image_shape
                        )
                    )
                    batch_size = len(boxes_torch)
                if points is not None:
                    point, point_label = points[i], point_labels[i]  # type: ignore
                    if point.ndim == 1:
                        point = point[None, None, :]  # [1, 1, 2]
                        point_label = np.asarray(point_label).reshape(1, -1)  # [1, 1]
                    elif point.ndim == 2:
                        point = point[None, ...]  # [1, N, 2]
                        point_label = np.asarray(point_label).reshape(1, -1)  # [1, N]
                    input_dict["point_coords"] = (
                        self.resize_transform.apply_coords_torch(
                            torch.as_tensor(point, device=self.device), image_shape
                        )
                    )
                    input_dict["point_labels"] = torch.as_tensor(
                        point_label, device=self.device
                    )
                    assert batch_size == -1 or batch_size == len(point) == len(
                        point_label
                    ), (
                        f"Mismatch batch_size {batch_size = } "
                        f"{len(point) = } {len(point_label) = }"
                    )

                output = self.model([input_dict], multimask_output=False)[0]

                mask = output["masks"].squeeze(1)
                mask = mask.cpu().numpy() if return_on_cpu else mask
                pred_iou = output["iou_predictions"].cpu().numpy()[:, 0]
                # output["low_res_logits"] has shape [n_bbox, 1, 256, 256]

                if not multiple_images:  # single input image
                    return (
                        (mask[0], pred_iou[0]) if squeeze_return else (mask, pred_iou)
                    )

                masks.append(mask)
                pred_ious.append(pred_iou)

        return masks, pred_ious
