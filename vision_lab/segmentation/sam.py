# python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
import os
from pathlib import Path
from typing import List, Tuple, Union

import gdown
import numpy as np
import torch
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

    def __init__(
        self,
        model_variant="vit_h",
        device="cuda",
    ):
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
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        boxes: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
        return_on_cpu=False,
        verbose=False,
    ) -> Union[
        Tuple[torch.Tensor, Union[np.float32, np.ndarray]],
        Tuple[List[torch.Tensor], List[np.ndarray]],
    ]:
        """
        :param images: Input RGB images, can be
                       a string path, a list of string paths,
                       a [H, W, 3] np.uint8 np.ndarray,
                       a list of [H, W, 3] np.uint8 np.ndarray,
                       a [n_images, H, W, 3] np.uint8 np.ndarray
        :param boxes: (n_images) list of pred_bbox as XYXY pixel coordinates
                      [n_bbox, 4] torch.float32 cuda Tensor
        :param return_on_cpu: whether to return masks as cuda Tensor or numpy array
        :param verbose: whether to print debug info
        :return masks: (n_images) list of predicted mask
                       [n_bbox, H, W] or [H, W] torch.bool cuda Tensor
        :return pred_ious: (n_images) list of [n_bbox,] or () np.float32 np.ndarray
        """
        masks, pred_ious = [], []

        with torch.cuda.device(self.device):
            # Process images and boxes
            images, is_list = load_image_arrays(images)
            if is_single_box := not isinstance(boxes, list):
                is_single_box = boxes.ndim == 1  # boxes has shape [4,]
                boxes = [boxes]
            assert len(images) == len(boxes), f"{len(images) = } {len(boxes) = }"

            # run SAM model on 1-image batch (on 11GB GPU)
            for image, box in zip(images, boxes):
                image_shape = image.shape[:2]

                processed_image = self.resize_transform.apply_image(image)
                processed_image = (
                    torch.as_tensor(processed_image, device=self.device)
                    .permute(2, 0, 1)
                    .contiguous()
                )

                output = self.model(
                    [
                        {
                            "image": processed_image,
                            "boxes": self.resize_transform.apply_boxes_torch(
                                torch.as_tensor(box, device=self.device), image_shape
                            ),
                            "original_size": image_shape,
                        }
                    ],
                    multimask_output=False,
                )[0]

                mask = output["masks"].squeeze(1)
                mask = mask.cpu().numpy() if return_on_cpu else mask
                pred_iou = output["iou_predictions"].cpu().numpy()[:, 0]
                # output["low_res_logits"] has shape [n_bbox, 1, 256, 256]

                if not is_list:
                    return (mask[0], pred_iou[0]) if is_single_box else (mask, pred_iou)

                masks.append(mask)
                pred_ious.append(pred_iou)

        return masks, pred_ious
