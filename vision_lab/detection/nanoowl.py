# https://github.com/xuanlinli17/nanoowl
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import os
import torch
from nanoowl.owl_predictor import OwlPredictor

from ..utils.io import load_image_arrays
from ..utils.timer import timer


class NanoOwl:
    """Nano Owl-ViT for object detection"""

    VARIANTS = [
        "google/owlvit-base-patch32",
        "google/owlvit-base-patch16",
        "google/owlvit-large-patch14",
        "google/owlv2-base-patch16-ensemble",
        "google/owlv2-large-patch14-ensemble",
    ]

    CKPT_DIR = Path(os.getenv("NANOOWL_TENSORRT_CKPT_DIR", Path.home() / "checkpoints/NanoOwl_TensorRT"))
    CKPT_PATHS = {
        "google/owlvit-base-patch32": CKPT_DIR / "owl_image_encoder_base_patch32.engine",
        "google/owlvit-base-patch16": CKPT_DIR / "owl_image_encoder_base_patch16.engine",
        "google/owlvit-large-patch14": CKPT_DIR / "owl_image_encoder_large_patch14.engine",
        "google/owlv2-base-patch16-ensemble": CKPT_DIR / "owlv2_image_encoder_base_patch16.engine",
        "google/owlv2-large-patch14-ensemble": CKPT_DIR / "owlv2_image_encoder_large_patch14.engine",
    }

    def __init__(
        self,
        model_variant="google/owlv2-large-patch14-ensemble",
        box_threshold=0.2,
        nms_threshold=0.3,
        max_image_batch_size=1,
        device="cuda",
    ):
        """
        :param box_threshold: filtering threshold for bbox
        :param nms_threshold: NMS IoU threshold (1.0 is no NMS)
        """
        assert model_variant in self.VARIANTS, f"Unknown {model_variant = }"
        self.model_variant = model_variant

        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.max_image_batch_size = max_image_batch_size
        self.device = device

        self.prompts = None
        self.text_encodings = None

        self.load_model()

    @timer
    def load_model(self):
        ckpt_path = self.CKPT_PATHS[self.model_variant]
        assert os.path.exists(ckpt_path), f"Missing {ckpt_path = }"
        self.model = OwlPredictor(
            self.model_variant,
            device=self.device,
            image_encoder_engine=ckpt_path,
            image_encoder_engine_max_batch_size=self.max_image_batch_size,
            no_roi_align=True,
        )

    @timer
    @torch.no_grad()
    def initialize_prompts(self, prompts: List[str]):
        self.prompts = prompts
        self.text_encodings = self.model.encode_text(prompts)

    @timer
    @torch.no_grad()
    def __call__(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        prompts: Optional[List[str]]=None,
        return_on_cpu=False,
        verbose=False,
    ) -> Union[
        Tuple[torch.Tensor, np.ndarray, np.ndarray],
        Tuple[List[torch.Tensor], List[np.ndarray], List[np.ndarray]],
    ]:
        """
        :param images: Input RGB images, can be
                       a string path, a list of string paths,
                       a [H, W, 3] np.uint8 np.ndarray,
                       a list of [H, W, 3] np.uint8 np.ndarray,
                       a [n_images, H, W, 3] np.uint8 np.ndarray
        :param prompts: a list of text detection prompts, same for all n_images
                        ["mug", "mug handle"]
        :param return_on_cpu: whether to return boxes as cuda Tensor or numpy array
        :param verbose: whether to print debug info
        :return boxes: (n_images) list of pred_bbox as XYXY pixel coordinates
                       [n_bbox, 4] torch.float32 cuda Tensor
        :return pred_indices: (n_images) list of [n_bbox,] integer np.ndarray
        :return pred_scores: (n_images) list of [n_bbox,] np.float32 np.ndarray
        """
        if prompts is None:
            assert self.prompts is not None, "Missing prompts"
        if prompts is not None:
            if (self.prompts is None) or (len(prompts) != len(self.prompts)) or (not all(a == b for a, b in zip(prompts, self.prompts))):
                self.initialize_prompts(prompts)

        if isinstance(images, np.ndarray) and images.ndim == 4:
            # [n_images, H, W, 3]
            images_arr = images
            is_list = True
        else:
            images, is_list = load_image_arrays(images) # List[np.ndarray], bool
            images_arr = np.stack(images) # [n_images, H, W, 3] np.uint8
        assert images_arr.shape[0] <= self.max_image_batch_size, f"Too many images {images_arr.shape[0]} > {self.max_image_batch_size}"

        output = self.model.predict(
            image=images_arr, 
            text=self.prompts, 
            text_encodings=self.text_encodings,
            threshold=self.box_threshold,
            nms_threshold=self.nms_threshold,
            pad_square=False
        )
        boxes, pred_indices, pred_scores, input_indices = (
            output.boxes, output.labels.cpu().numpy(), output.scores.cpu().numpy(), output.input_indices.cpu().numpy()
        )
        if return_on_cpu:
            boxes = boxes.cpu().numpy()

        ret_boxes, ret_pred_indices, ret_pred_scores = [], [], []
        for i in range(len(images)):
            indices = (input_indices == i)
            ret_boxes.append(boxes[indices])
            ret_pred_indices.append(pred_indices[indices])
            ret_pred_scores.append(pred_scores[indices])

        if not is_list:
            # remove batch dimension
            ret_boxes, ret_pred_indices, ret_pred_scores = ret_boxes[0], ret_pred_indices[0], ret_pred_scores[0]
        
        return ret_boxes, ret_pred_indices, ret_pred_scores
