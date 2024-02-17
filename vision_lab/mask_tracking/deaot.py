from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from seg_and_track_anything import SegTracker, aot_args, segtracker_args

from ..utils.io import load_image_arrays
from ..utils.timer import timer


class DeAoT:
    """DeAoT for object mask tracking"""

    CHECKPOINTS = {
        "r50_deaotl": "models/R50_DeAOTL_PRE_YTB_DAV.pth",
        "swinb_deaotl": "models/SwinB_DeAOTL_PRE_YTB_DAV.pth",
    }

    def __init__(
        self,
        root_path: Union[str, Path] = "/rl_benchmark/seg-and-track-anything",
        model_variant="swinb_deaotl",
        aot_max_len_long_term=10,
        num_trackers=1,
        device="cuda",
    ):
        """
        :param aot_max_len_long_term: maximum number of long-term key frames
        :param num_trackers: number of mask trackers (same as input batch_size)
        """
        root_path = Path(root_path)
        self.checkpoint = root_path / self.CHECKPOINTS[model_variant]
        self.model_variant = model_variant

        self.tracker_args = segtracker_args
        self.aot_args = aot_args

        self.aot_args["model"] = self.model_variant
        self.aot_args["model_path"] = self.checkpoint
        self.aot_args["max_len_long_term"] = aot_max_len_long_term
        if ":" in device:
            self.aot_args["gpu_id"] = int(device.split(":")[-1])

        self.device = device

        self.load_model(num_trackers)
        self.reset()

    @timer
    def load_model(self, num_trackers=1):
        self.trackers = [SegTracker(self.tracker_args, self.aot_args)]
        self.trackers_frame_idx = [0]

        # Shared aot_model for several segtracker
        self.aot_args["shared_model"] = self.trackers[0].tracker.model

        # Create multiple trackers
        self._ensure_num_tracker(num_trackers)

        # self.model = self.model.to(self.device)
        # self.model.eval()

    def _ensure_num_tracker(self, num_trackers: int):
        if (num_new := num_trackers - len(self.trackers)) > 0:
            for i in range(num_new):
                self.trackers.append(SegTracker(self.tracker_args, self.aot_args))
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
            self.trackers[i].restart_tracker()
            self.trackers_frame_idx[i] = 0

    @timer
    @torch.no_grad()
    def __call__(
        self,
        frames: Union[str, List[str], np.ndarray, List[np.ndarray]],
        masks: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        return_on_cpu=False,
        verbose=False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        :param frames: Input RGB image frames, can be
                       a string path, a list of string paths,
                       a [H, W, 3] np.uint8 np.ndarray,
                       a list of [H, W, 3] np.uint8 np.ndarray,
                       a [batch_size, H, W, 3] np.uint8 np.ndarray
        :param masks: (batch_size) list of mask (object indices map), 0 is BG
                      [H, W] bool/np.uint8 np.ndarray
                      If provided, the masks are included with find_new_objs
        :param return_on_cpu: reserved, will always return np.ndarray
        :param verbose: whether to print debug info
        :return masks: (batch_size) list of tracked mask, 0 is BG
                       [H, W] np.uint8 np.ndarray
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
                if frame_idx == 0:
                    assert mask is not None, "Initial mask is None"
                    tracker.add_reference(frame, mask, frame_step=0)
                    pred_mask = mask.copy()
                elif mask is not None:  # include the mask with find_new_objs
                    pred_mask = tracker.track(frame, update_memory=False)
                    pred_mask += tracker.find_new_objs(pred_mask, mask)
                    tracker.add_reference(frame, pred_mask, frame_step=0)
                else:
                    pred_mask = tracker.track(frame, update_memory=True)

                self.trackers_frame_idx[i] += 1
                if not is_list:
                    return pred_mask
                pred_masks.append(pred_mask)

        return pred_masks
