from pathlib import Path
from typing import List, Tuple, Union

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from ..utils.io import load_image_pils
from ..utils.timer import timer


def preprocess_images(image_pils) -> Tuple[np.ndarray, torch.Tensor]:
    """GroundingDINO image preprocessing
    Returns np.ndarray of shape [n_images, H, W, 3] and
    Tensor of shape [n_images, 3, H, W]"""

    def _preprocess_image(image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    images = []
    processed_images = []
    for image_pil in image_pils:
        images.append(np.asarray(image_pil))
        processed_images.append(_preprocess_image(image_pil))
    return np.stack(images), torch.stack(processed_images)


def get_grounding_output(model, images, captions, box_threshold, text_threshold):
    for i, caption in enumerate(captions):
        _caption = caption.lower().strip()
        if not _caption.endswith("."):
            _caption = _caption + "."
        captions[i] = _caption

    with torch.no_grad():
        outputs = model(images, captions=captions)

    boxes_filt_batch, pred_indices_batch, pred_scores_batch = [], [], []
    for i in range(len(images)):
        caption = captions[i]
        caption_lst = caption.strip(".").split(". ")

        scores = outputs["pred_logits"].sigmoid()[i]  # (nq, 256)
        boxes = outputs["pred_boxes"][i]  # (nq, 4)

        # filter output
        filt_mask = scores.max(dim=1)[0] > box_threshold
        scores_filt = scores[filt_mask]  # num_filt, 256
        boxes_filt = boxes[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_indices, pred_scores = [], []
        for score, box in zip(scores_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                score > text_threshold, tokenized, tokenlizer
            )
            pred_indices.append(caption_lst.index(pred_phrase))
            pred_scores.append(score.max().item())

        boxes_filt_batch.append(boxes_filt)
        pred_indices_batch.append(np.array(pred_indices))
        pred_scores_batch.append(np.array(pred_scores, dtype=np.float32))

    return boxes_filt_batch, pred_indices_batch, pred_scores_batch


class GroundingDINO:
    """GroundingDINO for object detection"""

    # GroundingDINO-T (Swin-T backbone, without RefCOCO)
    # GroundingDINO-B (Swin-B backbone, with RefCOCO)
    CONFIGS = {
        "swin-t": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "swin-b": "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py",
    }
    CHECKPOINTS = {
        "swin-t": "models/groundingdino_swint_ogc.pth",
        "swin-b": "models/groundingdino_swinb_cogcoor.pth",
    }

    def __init__(
        self,
        model_variant="swin-b",
        box_threshold=0.3,
        text_threshold=0.25,
        device="cuda",
    ):
        """
        :param box_threshold: filtering threshold for bbox
        :param text_threshold: filtering threshold for text
        """
        raise NotImplementedError(
            "Download checkpoint/config locally: prev in /rl_benchmark/grounded-sam"
        )

        self.config = root_path / self.CONFIGS[model_variant]
        self.checkpoint = root_path / self.CHECKPOINTS[model_variant]
        self.model_variant = model_variant

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        self.load_model()

    @timer
    def load_model(self):
        args = SLConfig.fromfile(self.config)
        args.device = self.device
        self.model = build_model(args)

        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        load_res = self.model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        self.model = self.model.to(self.device)
        self.model.eval()

    @timer
    @torch.no_grad()
    def __call__(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        prompts: List[str],
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
        boxes_filt_batch, pred_indices_batch, pred_scores_batch = [], [], []

        with torch.cuda.device(self.device):
            # Preprocess images and prompts
            image_pils, is_list = load_image_pils(images)
            images, processed_images = preprocess_images(image_pils)

            N, H, W, C = images.shape

            # Expected format for GroundingDINO
            prompts = ". ".join(prompts) + "."
            if isinstance(prompts, str):
                prompts = [prompts] * len(image_pils)

            # run GroundingDINO model on 8-image batch (on 11GB GPU)
            bs = 8
            for i in range(int(np.ceil(len(images) / bs))):
                boxes_filt, pred_indices, pred_scores = get_grounding_output(
                    self.model,
                    processed_images[i * 8 : (i + 1) * 8].to(self.device),
                    prompts[i * 8 : (i + 1) * 8],
                    self.box_threshold,
                    self.text_threshold,
                )
                boxes_filt_batch.extend(boxes_filt)
                pred_indices_batch.extend(pred_indices)
                pred_scores_batch.extend(pred_scores)

            # Convert boxes_filt_batch from normalized coordinate to pixels
            for i, boxes_filt in enumerate(boxes_filt_batch):
                boxes_filt *= torch.Tensor([[W, H, W, H]]).to(self.device)
                boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
                boxes_filt[:, 2:] += boxes_filt[:, :2]
                boxes_filt_batch[i] = (
                    boxes_filt.cpu().numpy() if return_on_cpu else boxes_filt
                )

        if is_list:
            return boxes_filt_batch, pred_indices_batch, pred_scores_batch
        else:
            return boxes_filt_batch[0], pred_indices_batch[0], pred_scores_batch[0]
