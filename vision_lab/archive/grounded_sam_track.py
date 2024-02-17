import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Grounding DINO
import groundingdino.datasets.transforms as T

# import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image

# seg_and_track_anything
from seg_and_track_anything import SegTracker, aot_args, segtracker_args

# segment anything
from segment_anything import SamPredictor, build_sam, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from .utils.io import load_image_arrays, load_image_pils
from .utils.timer import RuntimeTimer

# REPO_ROOT
if Path("/kolin-fast").exists():
    _REPO_ROOT_ = Path("/kolin-fast/rl_benchmark")
else:
    _REPO_ROOT_ = Path("/rl_benchmark")


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


def get_grounding_output(
    model,
    images,
    captions,
    box_threshold,
    text_threshold,
    with_logits=True,
    device="cpu",
):
    for i, caption in enumerate(captions):
        _caption = caption.lower().strip()
        if not _caption.endswith("."):
            _caption = _caption + "."
        captions[i] = _caption

    # model = model.to(device)
    # images = images.to(device)
    with torch.no_grad():
        outputs = model(images, captions=captions)

    boxes_filt_batch, pred_phrases_batch = [], []
    for i in range(len(images)):
        caption = captions[i]

        logits = outputs["pred_logits"].cpu().sigmoid()[i]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[i]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        boxes_filt_batch.append(boxes_filt)
        pred_phrases_batch.append(pred_phrases)

    return boxes_filt_batch, pred_phrases_batch


class GroundedSAM:
    """Grounded Segment Anything (GroundingDINO + SAM)"""

    REPO_ROOT = _REPO_ROOT_ / "grounded-sam"

    # GroundingDINO-T (Swin-T backbone, without RefCOCO)
    # GroundingDINO-B (Swin-B backbone, with RefCOCO)
    GROUNDING_DINO_CONFIGS = {
        "swin-t": REPO_ROOT
        / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "swin-b": REPO_ROOT
        / "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py",
    }
    GROUNDING_DINO_CHECKPOINTS = {
        "swin-t": REPO_ROOT / "models/groundingdino_swint_ogc.pth",
        "swin-b": REPO_ROOT / "models/groundingdino_swinb_cogcoor.pth",
    }
    SAM_CHECKPOINTS = {
        "vit_b": REPO_ROOT / "models/sam_vit_b_01ec64.pth",
        "vit_l": REPO_ROOT / "models/sam_vit_l_0b3195.pth",
        "vit_h": REPO_ROOT / "models/sam_vit_h_4b8939.pth",
    }

    def __init__(
        self,
        grounding_dino_model_variant="swin-b",
        sam_model_variant="vit_h",
        box_threshold=0.3,
        text_threshold=0.25,
        device="cuda",
    ):
        raise DeprecationWarning("This module is deprecated.")
        self.grounding_dino_config = self.GROUNDING_DINO_CONFIGS[
            grounding_dino_model_variant
        ]
        self.grounding_dino_checkpoint = self.GROUNDING_DINO_CHECKPOINTS[
            grounding_dino_model_variant
        ]
        self.sam_checkpoint = self.SAM_CHECKPOINTS[sam_model_variant]

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        # Load grounding_dino_model and sam_model
        self.load_grounding_dino_model()
        self.load_sam_model(sam_model_variant)

    def load_grounding_dino_model(self):
        with RuntimeTimer("Load GroundingDINO model", enabled=True):
            args = SLConfig.fromfile(self.grounding_dino_config)
            args.device = self.device
            self.grounding_dino_model = build_model(args)

            checkpoint = torch.load(self.grounding_dino_checkpoint, map_location="cpu")
            load_res = self.grounding_dino_model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False
            )
            print(load_res)
            _ = self.grounding_dino_model.eval()
            self.grounding_dino_model = self.grounding_dino_model.to(self.device)

    def load_sam_model(self, model_variant="vit_h"):
        with RuntimeTimer("Load SAM model", enabled=True):
            self.sam_model = sam_model_registry[model_variant](
                checkpoint=self.sam_checkpoint
            ).to(self.device)
            self.sam_resize_transform = ResizeLongestSide(
                self.sam_model.image_encoder.img_size
            )

    def predict(
        self,
        input_images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        text_prompts: Union[str, List[str]],
        verbose=False,
    ) -> Tuple[List[np.ndarray], List[List[str]], List[np.ndarray]]:
        """input_images can be [n_images, H, W, 3]
        :return boxes_filt_batch: n_images [n_boxes, 4] np.ndarray
        :return pred_phrases_batch: n_images [n_boxes] list of pred_phrases str
        :return pred_masks_batch: n_images [n_boxes, 1, H, W] np.ndarray
        :return batched_output: n_images list of dictionary containing
                "masks": [n_boxes, 1, H, W] GPU Tensor
                "iou_predictions": [n_boxes, 1] GPU Tensor
                "low_res_logits": [n_boxes, 1, h, w] GPU Tensor
        """
        with torch.cuda.device(self.device):
            total_elapsed_time = 0.0

            with RuntimeTimer("Image preprocessing", verbose) as t:
                image_pils, _ = load_image_pils(input_images)
                images, processed_images = preprocess_images(image_pils)
                if isinstance(text_prompts, str):
                    text_prompts = [text_prompts] * len(image_pils)
            total_elapsed_time += t.elapsed_time

            # run grounding dino model
            with RuntimeTimer("GroundingDINO", verbose) as t:
                boxes_filt_batch, pred_phrases_batch = [], []

                # run SAM model on 8-image batch (on 11GB GPU)
                bs = 8
                for i in range(int(np.ceil(len(images) / bs))):
                    boxes_filt, pred_phrases = get_grounding_output(
                        self.grounding_dino_model,
                        processed_images[i * 8 : (i + 1) * 8].to(self.device),
                        text_prompts[i * 8 : (i + 1) * 8],
                        self.box_threshold,
                        self.text_threshold,
                        device=self.device,
                    )
                    boxes_filt_batch.extend(boxes_filt)
                    pred_phrases_batch.extend(pred_phrases)
            total_elapsed_time += t.elapsed_time

            # run SAM model
            with RuntimeTimer("SAM prediction", verbose) as t:
                batched_output = []

                # run SAM model on 1-image batch (on 11GB GPU)
                for image, boxes_filt in zip(images, boxes_filt_batch):
                    processed_image = self.sam_resize_transform.apply_image(image)
                    processed_image = (
                        torch.as_tensor(processed_image, device=self.device)
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    H, W, C = image.shape
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    output = self.sam_model(
                        [
                            {
                                "image": processed_image,
                                "boxes": self.sam_resize_transform.apply_boxes_torch(
                                    boxes_filt.to(self.device), image.shape[:2]
                                ),
                                "original_size": image.shape[:2],
                            }
                        ],
                        multimask_output=False,
                    )[0]

                    # Copy from GPU to CPU
                    output = {k: v.cpu() for k, v in output.items()}
                    batched_output.append(output)
                pred_masks_batch = [d["masks"] for d in batched_output]
            total_elapsed_time += t.elapsed_time

            if verbose:
                print(
                    f"Inferencing {len(batched_output)} images took "
                    f"{total_elapsed_time:.3f} seconds"
                )

        # Convert torch.Tensor to np.ndarray
        boxes_filt_batch = [b.numpy() for b in boxes_filt_batch]
        pred_masks_batch = [m.numpy() for m in pred_masks_batch]

        return boxes_filt_batch, pred_phrases_batch, pred_masks_batch

    @staticmethod
    def merge_pred_masks(
        pred_phrases_batch: List[List[str]],
        pred_masks_batch: List[np.ndarray],
        object_labels_batch: Union[List[str], List[List[str]]],
        extra_masks_batch: List[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        :param pred_phrases_batch: n_images [n_boxes] list of pred_phrases str
        :param pred_masks_batch: n_images [n_boxes, 1, H, W] bool np.ndarray
        :param object_labels_batch: n_images list of object_label str
        :param extra_masks_batch: n_images [H, W] bool np.ndarray
        :return pred_masks_pruned: n_images [H, W] list of np.uint8 np.ndarray
                                   mask value is index+1 in object_labels
        """

        def find_most_confident_label(pred_phrases, label):
            pred_label_idx = None
            best_logit = 0.0
            for i, pred_phrase in enumerate(pred_phrases):
                pred_label, pred_logit = re.split("\(|\)", pred_phrase)[:2]
                if label.lower() in pred_label and float(pred_logit) > best_logit:
                    pred_label_idx = i
                    best_logit = float(pred_logit)
            return pred_label_idx

        num_images = len(pred_phrases_batch)
        if isinstance(object_labels_batch[0], str):
            object_labels_batch = [object_labels_batch] * num_images
        if extra_masks_batch is None:
            extra_masks_batch = [None] * num_images
        assert (
            len(pred_phrases_batch)
            == len(pred_masks_batch)
            == len(object_labels_batch)
            == len(extra_masks_batch)
        ), (
            f"{len(pred_phrases_batch)} pred_phrases"
            f" != {len(pred_masks_batch)} pred_masks"
            f" != {len(object_labels_batch)} object_labels"
            f" != {len(extra_masks_batch)} extra_masks"
        )

        pred_masks_pruned = []

        for img_i, pred_phrases in enumerate(pred_phrases_batch):
            pred_masks = pred_masks_batch[img_i]
            object_labels = object_labels_batch[img_i]
            extra_mask = extra_masks_batch[img_i]

            pred_masks_pruned.append(np.zeros(pred_masks.shape[-2:], dtype=np.uint8))

            for label_i, object_label in enumerate(object_labels):
                pred_label_idx = find_most_confident_label(pred_phrases, object_label)
                if pred_label_idx is None:  # label not found
                    print(f"Label {object_label} not found")
                    continue

                pred_mask = pred_masks[pred_label_idx, 0]  # [H, W]
                if extra_mask is None:
                    pred_masks_pruned[-1][pred_mask] = label_i + 1
                else:
                    pred_masks_pruned[-1][extra_mask & pred_mask] = label_i + 1

        return pred_masks_pruned

    @staticmethod
    def save_pred_result(
        output_dir: Path,
        image: np.ndarray,
        text_prompt: str,
        boxes_filt: np.ndarray,
        pred_phrases: List[str],
        pred_masks: np.ndarray,
    ) -> None:
        """
        :param image: [H, W, C] np.uint8
        :param text_prompt: text prompt used to generate prediction
        :param boxes_filt: [n_boxes, 4] np.float32
        :param pred_phrases: [n_boxes] list of pred_phrases str
        :param pred_masks: [n_boxes, 1, H, W] np.bool
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        output_dir = str(output_dir)

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in pred_masks:
            show_mask(mask, plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box, plt.gca(), label)
        plt.title(text_prompt)

        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        save_mask_data(output_dir, pred_masks, boxes_filt, pred_phrases)


class GroundedSAMTrack(GroundedSAM):
    AOT_REPO_ROOT = _REPO_ROOT_ / "seg-and-track-anything"

    AOT_CHECKPOINTS = {
        "r50_deaotl": AOT_REPO_ROOT / "models/R50_DeAOTL_PRE_YTB_DAV.pth",
        "swinb_deaotl": AOT_REPO_ROOT / "models/SwinB_DeAOTL_PRE_YTB_DAV.pth",
    }

    def __init__(
        self,
        *args,
        aot_model_variant="r50_deaotl",
        aot_max_len_long_term=10,
        num_trackers=1,
        predict_gap=10,
        prompt_with_robot_arm=True,
        tracking_device=None,
        **kwargs,
    ):
        self.segtracker_args = segtracker_args
        self.aot_args = aot_args

        self.aot_checkpoint = self.AOT_CHECKPOINTS[aot_model_variant]
        self.aot_args["model"] = aot_model_variant
        self.aot_args["model_path"] = self.aot_checkpoint
        self.aot_args["max_len_long_term"] = aot_max_len_long_term

        self.predict_gap = predict_gap
        self.prompt_with_robot_arm = prompt_with_robot_arm

        super().__init__(*args, **kwargs)

        # Use separate device for mask tracking
        self.tracking_device = tracking_device
        if self.tracking_device is None:
            self.tracking_device = self.device

        # Load AOT model
        if ":" in self.tracking_device:
            self.aot_args["gpu_id"] = int(self.tracking_device.split(":")[-1])
        self.load_aot_model()
        # Shared aot_model for several segtracker
        self.aot_args["shared_model"] = self.segtrackers[0].tracker.model

        # Create multiple trackers
        self.ensure_num_segtracker(num_trackers)

    def load_aot_model(self):
        self.segtrackers = [SegTracker(self.segtracker_args, self.aot_args)]
        self.segtrackers[-1].restart_tracker()

    def ensure_num_segtracker(self, num_trackers: int):
        if (num_new := num_trackers - len(self.segtrackers)) > 0:
            for i in range(num_new):
                self.segtrackers.append(SegTracker(self.segtracker_args, self.aot_args))
                self.segtrackers[-1].restart_tracker()

    def _process_idx(self, idx):
        if idx is None:
            idx = np.arange(len(self.segtrackers))
        if isinstance(idx, int):
            idx = [idx]
        return idx

    def restart_tracker(self, idx=None):
        idx = self._process_idx(idx)
        for i in idx:
            self.segtrackers[i].restart_tracker()

    def add_reference(
        self,
        frames: Union[str, List[str], np.ndarray, List[np.ndarray]],
        masks: List[np.ndarray],
        frame_step=0,
        idx=None,
    ) -> None:
        """
        :param frames: can be a string path, a list of string paths,
                       a [H, W, 3] np.ndarray,
                       a list of [H, W, 3] np.ndarray,
                       a [n_images, H, W, 3] np.ndarray
        :param masks: predicted mask for the frames,
                      a list of [H, W] np.uint8 np.ndarray
        """
        with torch.cuda.device(self.tracking_device):
            frames = load_image_arrays(frames)

            idx = self._process_idx(idx)
            assert (
                len(frames) == len(masks) == len(idx)
            ), f"{len(frames)} frames != {len(masks)} masks != {len(idx)} idx"

            for i, frame, mask in zip(idx, frames, masks):
                self.segtrackers[i].add_reference(frame, mask, frame_step)

    def track(
        self,
        frames: Union[str, List[str], np.ndarray, List[np.ndarray]],
        update_memory=False,
        idx=None,
    ) -> List[np.ndarray]:
        """
        :param frames: can be a string path, a list of string paths,
                       a [H, W, 3] np.ndarray,
                       a list of [H, W, 3] np.ndarray,
                       a [n_images, H, W, 3] np.ndarray
        :return masks: predicted mask for the frames,
                       a list of [H, W] np.uint8 np.ndarray
        """
        with torch.cuda.device(self.tracking_device):
            frames = load_image_arrays(frames)

            idx = self._process_idx(idx)
            assert len(frames) == len(idx), f"{len(frames)} frames != {len(idx)} idx"

            pred_masks = []
            for i, frame in zip(idx, frames):
                pred_masks.append(self.segtrackers[i].track(frame, update_memory))

        return pred_masks

    def find_new_objs(
        self, track_masks: List[np.ndarray], seg_masks: List[np.ndarray], idx=None
    ) -> List[np.ndarray]:
        """Find new objects from tracking masks and segmentation masks
        :param track_masks: a list of [H, W] np.uint8 np.ndarray
        :param seg_masks: a list of [H, W] np.uint8 np.ndarray
        :return new_masks: a list of [H, W] np.uint8 np.ndarray
        """
        with torch.cuda.device(self.tracking_device):
            idx = self._process_idx(idx)
            assert len(track_masks) == len(seg_masks) == len(idx), (
                f"{len(track_masks)} track_masks"
                f" != {len(seg_masks)} seg_masks != {len(idx)} idx"
            )

            new_masks = []
            for i, track_mask, seg_mask in zip(idx, track_masks, seg_masks):
                new_masks.append(
                    track_mask + self.segtrackers[i].find_new_objs(track_mask, seg_mask)
                )
        return new_masks

    def predict_and_track_batch(
        self,
        batch_frames: Union[str, List[str], np.ndarray, List[np.ndarray]],
        batch_frames_i: np.ndarray,
        object_texts: List[str],
        batch_extra_frame_masks: List[np.ndarray] = None,
        tracker_idx=None,
    ) -> Dict[str, List[np.ndarray]]:
        """Predict and track objects given object_texts based on frame_i
        :param batch_frames: can be a string path, a list of string paths,
                             a [H, W, 3] np.ndarray,
                             a list of [H, W, 3] np.ndarray,
                             a [n_images, H, W, 3] np.ndarray
        :param batch_frames_i: a [n_images, 1] np.int np.ndarray
                               zero-indexed frame indices
        :param object_texts: a list of text describing objects
                             same for the entire batch_frames
        :param batch_extra_frame_masks: n_images [H, W] bool np.ndarray
        :param tracker_idx: array of tracker indices, matching batch_frames
        :return pred_masks: predicted mask for previou-step frames,
                            a list of n_images [H, W] np.uint8 np.ndarray
                            mask value is index+1 in object_texts
        :return pred_phrases: a list of n_images [n_boxes] list of pred_phrases
        :return boxes_filt: a list of n_images [n_boxes, 4] np.ndarray
        """
        batch_frames = load_image_arrays(batch_frames)
        batch_frames_i = np.array(batch_frames_i).astype(int)
        if batch_extra_frame_masks is None:
            batch_extra_frame_masks = [None] * len(batch_frames)
        tracker_idx = np.array(self._process_idx(tracker_idx))
        assert (
            len(batch_frames)
            == len(batch_frames_i)
            == len(batch_extra_frame_masks)
            == len(tracker_idx)
        ), (
            f"{len(batch_frames)} frames"
            f" != {len(batch_frames_i)} frames_i"
            f" != {len(batch_extra_frame_masks)} extra_frame_masks"
            f" != {len(tracker_idx)} tracker_idx"
        )

        text_prompt = ". ".join(object_texts) + "."
        if self.prompt_with_robot_arm:
            text_prompt += " robot arm."

        # For batch_frames, get their mask generation type
        #   Below are indices into batch_frames
        init_pred_i = np.flatnonzero(batch_frames_i == 0)
        pred_gap_i = np.flatnonzero(batch_frames_i % self.predict_gap == 0)
        pred_gap_i = np.setdiff1d(pred_gap_i, init_pred_i)
        track_i = np.setdiff1d(
            np.setdiff1d(np.arange(len(batch_frames_i)), init_pred_i), pred_gap_i
        )

        ret = {
            "pred_masks": [None] * len(batch_frames),
            "pred_phrases": [None] * len(batch_frames),
            "boxes_filt": [None] * len(batch_frames),
        }
        # Initial prediction
        if len(init_pred_i) > 0:
            # reset tracker for init_pred_i
            self.restart_tracker(tracker_idx[init_pred_i])

            input_images = [batch_frames[i] for i in init_pred_i]
            boxes_filt_batch, pred_phrases_batch, pred_masks_batch = self.predict(
                input_images, text_prompt
            )
            # Find most confident mask and merge pred_masks
            pred_masks_batch = self.merge_pred_masks(
                pred_phrases_batch,
                pred_masks_batch,
                object_texts,
                [batch_extra_frame_masks[i] for i in init_pred_i],
            )
            self.add_reference(
                input_images, pred_masks_batch, idx=tracker_idx[init_pred_i]
            )
            for i in init_pred_i:
                ret["pred_masks"][i] = pred_masks_batch[i]
                ret["pred_phrases"][i] = pred_phrases_batch[i]
                ret["boxes_filt"][i] = boxes_filt_batch[i]

        # Intermediate prediction per predict_gap frames
        if len(pred_gap_i) > 0:
            input_images = [batch_frames[i] for i in pred_gap_i]
            boxes_filt_batch, pred_phrases_batch, pred_masks_batch = self.predict(
                input_images, text_prompt
            )
            # Find most confident mask and merge pred_masks
            seg_masks_batch = self.merge_pred_masks(
                pred_phrases_batch,
                pred_masks_batch,
                object_texts,
                [batch_extra_frame_masks[i] for i in pred_gap_i],
            )
            track_masks_batch = self.track(input_images, idx=tracker_idx[pred_gap_i])
            pred_masks_batch = self.find_new_objs(
                track_masks_batch, seg_masks_batch, idx=tracker_idx[pred_gap_i]
            )
            self.add_reference(
                input_images, pred_masks_batch, idx=tracker_idx[pred_gap_i]
            )
            for i in pred_gap_i:
                ret["pred_masks"][i] = pred_masks_batch[i]
                ret["pred_phrases"][i] = pred_phrases_batch[i]
                ret["boxes_filt"][i] = boxes_filt_batch[i]

        # Track masks
        if len(track_i) > 0:
            input_images = [batch_frames[i] for i in track_i]
            pred_masks_batch = self.track(
                input_images, update_memory=True, idx=tracker_idx[track_i]
            )
            for i in track_i:
                ret["pred_masks"][i] = pred_masks_batch[i]

        return ret


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def show_mask(mask: np.ndarray, ax, random_color=False) -> None:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box: np.ndarray, ax, label) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(
    output_dir,
    mask_list: Union[torch.Tensor, np.ndarray],
    box_list: Union[torch.Tensor, np.ndarray],
    label_list: List[str],
) -> None:
    value = 0  # 0 for background

    mask_img = np.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_img[mask[0]] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img)
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "mask.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )

    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        if isinstance(box, torch.Tensor):
            box = box.numpy()
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            "value": value,
            "label": name,
            "logit": float(logit),
            "box": box.tolist(),
        })
    with open(os.path.join(output_dir, "mask.json"), "w") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--input_image", type=str, required=True, help="path to image file"
    )
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="outputs",
        required=True,
        help="output directory",
    )

    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="running on cpu only!, default=False"
    )
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, _ = load_image_pils(image_path)[0]
    images, processed_images = preprocess_images([image_pil])
    image, processed_image = images[0], processed_images[0]
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device).to(device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, processed_images.to(device), [text_prompt], box_threshold, text_threshold
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    H, W, C = image.shape
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    )

    pred_masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in pred_masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.title(text_prompt)

    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )

    save_mask_data(output_dir, pred_masks, boxes_filt, pred_phrases)
