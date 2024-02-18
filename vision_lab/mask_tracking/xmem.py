from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from XMem.inference.inference_core import InferenceCore
from XMem.model.network import XMem as XMemNetwork

from ..utils.io import load_image_arrays
from ..utils.timer import timer


class XMem:
    """XMem for object mask tracking"""

    CHECKPOINTS = {
        "default": "saves/XMem.pth",
        "s012": "saves/XMem-s012.pth",
    }

    def __init__(
        self,
        model_variant="default",
        top_k=30,
        mem_every=5,
        deep_update_every=-1,
        enable_long_term=True,
        enable_long_term_count_usage=True,
        num_prototypes=128,
        min_mid_term_frames=5,
        max_mid_term_frames=10,
        max_long_term_elements=10000,
        num_trackers=1,
        device="cuda",
    ):
        """
        :param top_k: top-k filtering
        :param mem_every: Update working memory (short-term) every r-th frame.
                          r in paper. Increase to improve running speed.
        :param deep_update_every: Leave -1 normally to synchronize with mem_every
        :param enable_long_term: Enable long-term memory
        :param enable_long_term_count_usage: Count usage of long-term memory
        :param num_prototypes: P in paper, proportional to the size of long-term memory
        :param min_mid_term_frames: Keep 1st frame and the most recent T_min -1 frames
                                        in working memory
                                    T_min in paper, decrease to save memory
        :param max_mid_term_frames: Maximum size of working memory (short-term)
                                    T_max in paper, decrease to save memory
        :param max_long_term_elements: LT_max in paper, increase if objects
                                       disappear for a long time
        :param num_trackers: number of mask trackers (same as input batch_size)
        """
        raise NotImplementedError(
            "Download checkpoint locally: prev in /rl_benchmark/XMem"
        )

        self.checkpoint = root_path / self.CHECKPOINTS[model_variant]
        self.model_variant = model_variant

        self.config = dict(
            top_k=top_k,
            mem_every=mem_every,
            deep_update_every=deep_update_every,
            enable_long_term=enable_long_term,
            enable_long_term_count_usage=enable_long_term_count_usage,
            num_prototypes=num_prototypes,
            min_mid_term_frames=min_mid_term_frames,
            max_mid_term_frames=max_mid_term_frames,
            max_long_term_elements=max_long_term_elements,
        )

        self.device = device

        assert num_trackers == 1, "num_trackers > 1 is not yet implemented"

        self.load_model(num_trackers)
        self.reset()

    @timer
    def load_model(self, num_trackers=1):
        self.network = XMemNetwork(self.config, self.checkpoint)
        self.network = self.network.to(self.device).eval()

        self.trackers = [InferenceCore(self.network, config=self.config)]
        self.trackers_frame_idx = [0]

        # Create multiple trackers
        self._ensure_num_tracker(num_trackers)

        self.image_normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _ensure_num_tracker(self, num_trackers: int):
        if (num_new := num_trackers - len(self.trackers)) > 0:
            for i in range(num_new):
                raise NotImplementedError("num_trackers > 1 is not yet implemented")
                self.trackers.append(InferenceCore(self.network, config=self.config))
                self.trackers_frame_idx.append(0)

    def _process_idx(self, idx: Optional[Union[int, List[int]]] = None) -> List[int]:
        if idx is None:
            idx = list(range(len(self.trackers)))
        if isinstance(idx, int):
            idx = [idx]
        return idx

    def reset(self, idx: Optional[Union[int, List[int]]] = None):
        """Reset the mask tracker"""
        for i in self._process_idx(idx):
            self.trackers[i].clear_memory()
            self.trackers[i].set_all_labels(None)
            self.trackers_frame_idx[i] = 0

    @timer
    @torch.no_grad()
    def __call__(
        self,
        frames: Union[str, List[str], np.ndarray, List[np.ndarray]],
        masks: Optional[Union[List[torch.Tensor], List[np.ndarray]]] = None,
        return_on_cpu=False,
        verbose=False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        :param frames: Input RGB image frames, can be
                       a string path, a list of string paths,
                       a [H, W, 3] np.uint8 np.ndarray,
                       a list of [H, W, 3] np.uint8 np.ndarray,
                       a [batch_size, H, W, 3] np.uint8 np.ndarray
        :param masks: (batch_size) list of mask (object indices map), 0 is BG
                      [H, W] torch.bool/torch.uint8 cuda Tensor
                      If provided, the masks are included directly
        :param return_on_cpu: whether to return boxes as cuda Tensor or numpy array
        :param verbose: whether to print debug info
        :return masks: (batch_size) list of tracked mask, 0 is BG
                       [H, W] torch.uint8 cuda Tensor
        """
        pred_masks = []

        with torch.cuda.device(self.device):
            # Process frames and masks
            frames, is_list = load_image_arrays(frames)
            if masks is None:
                masks = [None] * len(frames)
            elif not isinstance(masks, list):
                masks = [masks]
            assert (
                len(frames) == len(masks) == len(self.trackers)
            ), f"{len(frames) = } {len(masks) = } {len(self.trackers) = }"

            for i, (frame, mask, tracker, frame_idx) in enumerate(
                zip(frames, masks, self.trackers, self.trackers_frame_idx)
            ):
                # Convert frame: [H, W, 3] => [3, H, W]
                frame = self.image_normalization(
                    torch.as_tensor(frame, device=self.device).permute(2, 0, 1) / 255.0
                )
                # Convert mask format [H, W] => [n_obj, H, W]
                if mask is not None:
                    mask = (
                        F.one_hot(torch.as_tensor(mask, device=self.device).long())
                        .permute(2, 0, 1)[1:]
                        .float()
                    )

                if frame_idx == 0:
                    assert mask is not None, "Initial mask is None"
                    labels = list(range(1, mask.shape[0] + 1))
                    tracker.set_all_labels(labels)
                    pred_prob = tracker.step(frame, mask, labels)
                elif mask is not None:  # include the mask directly
                    labels = list(range(1, mask.shape[0] + 1))
                    tracker.set_all_labels(labels)
                    pred_prob = tracker.step(frame, mask, labels)
                else:
                    pred_prob = tracker.step(frame)

                self.trackers_frame_idx[i] += 1
                # pred_prob has shape [n_obj+1, H, W]
                pred_mask = pred_prob.argmax(0).to(torch.uint8)
                pred_mask = pred_mask.cpu().numpy() if return_on_cpu else pred_mask
                if not is_list:
                    return pred_mask
                pred_masks.append(pred_mask)

        return pred_masks
