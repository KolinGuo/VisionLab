# https://github.com/xuanlinli17/efficientvit
# https://github.com/xuanlinli17/efficientvit/blob/master/applications/sam.md
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gdown
import numpy as np
import torch
from real_robot.utils.logger import get_logger
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # type: ignore

from ..utils.io import load_image_arrays
from ..utils.timer import timer


class EfficientViT_SAM:
    """SAM for object segmentation"""

    CKPT_DIR = Path(os.getenv("EFFICIENTVIT_SAM_CKPT_DIR", Path.home() / "checkpoints/EfficientViT_SAM"))

    CKPT_PATHS = {
        "l0": CKPT_DIR / "l0.pt",
        "l1": CKPT_DIR / "l1.pt",
        "l2": CKPT_DIR / "l2.pt",
        "xl0": CKPT_DIR / "xl0.pt",
        "xl1": CKPT_DIR / "xl1.pt",
    }

    CKPT_GDOWN_PARAMS = {
        "l0": {
            "url": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l0.pt",
        },
        "l1": {
            "url": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l1.pt",
        },
        "l2": {
            "url": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt",
        },
        "xl0": {
            "url": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl0.pt",
        },
        "xl1": {
            "url": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt",
        },
    }
    logger = get_logger("EfficientViT_SAM")

    def __init__(
        self,
        model_variant="xl0",
        device="cuda",
    ):
        self.logger.info('Using EfficientViT-SAM model variant: "%s"', model_variant)

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

        self.model = create_sam_model(self.model_variant, True, self.ckpt_path).to(self.device).eval()
        self.predictor = EfficientViTSamPredictor(self.model)

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

            # run EfficientViT-SAM model on 1-image batch
            for i, image in enumerate(images):
                self.predictor.set_image(image)

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
                    point = self.predictor.apply_coords(point)
                    point_torch = torch.as_tensor(point, dtype=torch.float, device=self.device)
                    point_label_torch = torch.as_tensor(point_label, dtype=torch.int, device=self.device)
                    mask, pred_iou, _ = self.predictor.predict_torch(
                        point_coords=point_torch,
                        point_labels=point_label_torch,
                        multimask_output=False,
                    )
                elif boxes is not None:
                    box = boxes[i]
                    if box.ndim == 1:
                        box = box[None, :]  # [1, 4]
                    box = self.predictor.apply_boxes(box)
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
                    mask, pred_iou, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=box_torch,
                        multimask_output=False,
                    )

                mask = mask.squeeze(1) # [1, 1, H, W] -> [1, H, W]
                mask = mask.cpu().numpy() if return_on_cpu else mask
                pred_iou = pred_iou.cpu().numpy()[:, 0] # [1, 1] -> [1]

                if not multiple_images:  # single input image
                    return (
                        (mask[0], pred_iou[0]) if squeeze_return else (mask, pred_iou)
                    )

                masks.append(mask)
                pred_ious.append(pred_iou)

        return masks, pred_ious
