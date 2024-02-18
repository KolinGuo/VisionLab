import os
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import ruamel.yaml
import torch
from BundleSDF import BundleSdf
from BundleSDF.utils import set_seed
from real_robot.utils.logger import get_logger

from ..utils.timer import timer


class BundleSDF:
    """BundleSDF for object pose tracking"""

    LOFTR_CHECKPOINTS = {
        "default": "BundleSDF/BundleTrack/LoFTR/weights/outdoor_ds.ckpt",
    }

    def __init__(
        self,
        model_variant="default",
        run_name="debug_run",
        feature_size=256,
        use_nerf=True,
        run_nerf_async=True,
        start_nerf_keyframes=5,
        verbose_level=0,
        save_results=False,
        use_gui=False,
        device="cuda",
    ):
        """
        :param run_name: directory name under root_path for saving debug info.
        :param feature_size: image size used for feature extraction
        :param start_nerf_keyframes: start training NeRF after this many keyframes
        :param verbose_level: logging verbose level for BundleTrack (max=1)
        :param save_results: save results to debug_dir (might need verbose_level == 2)
        :param use_gui: whether to show gui
        """
        raise NotImplementedError(
            "Download checkpoint/config locally: prev in /rl_benchmark/BundleSDF"
        )

        self.checkpoint = root_path / self.LOFTR_CHECKPOINTS[model_variant]
        self.model_variant = model_variant
        # Create output directory
        self.output_dir = (
            Path(os.getenv("REAL_ROBOT_LOG_DIR", root_path / "logs")) / "BundleSDF"
        )
        shutil.rmtree(self.output_dir, ignore_errors=True)
        self.output_dir.mkdir(parents=True)

        # Read default config
        yaml = ruamel.yaml.YAML()
        self.track_cfg_path = root_path / "config_track.yml"
        self.track_config = yaml.load(self.track_cfg_path.open("r"))
        self.nerf_cfg_path = root_path / "config_nerf.yml"
        self.nerf_config = yaml.load(self.nerf_cfg_path.open("r"))

        # Update config
        self.track_config["SPDLOG"] = verbose_level
        self.track_config["depth_processing"]["zfar"] = 1
        self.track_config["depth_processing"]["percentile"] = 95
        self.track_config["erode_mask"] = 3
        self.track_config["debug_dir"] = debug_dir = str(self.output_dir / "debug")
        self.track_config["bundle"]["max_BA_frames"] = 10
        self.track_config["bundle"]["max_optimized_feature_loss"] = 0.03
        self.track_config["feature_corres"]["max_dist_neighbor"] = 0.02
        self.track_config["feature_corres"]["max_normal_neighbor"] = 30
        self.track_config["feature_corres"]["max_dist_no_neighbor"] = 0.01
        self.track_config["feature_corres"]["max_normal_no_neighbor"] = 20
        self.track_config["feature_corres"]["map_points"] = True
        self.track_config["feature_corres"]["resize"] = feature_size
        self.track_config["feature_corres"]["rematch_after_nerf"] = (
            True  # TODO: what is this
        )
        self.track_config["keyframe"]["min_rot"] = 5
        self.track_config["ransac"]["inlier_dist"] = 0.01
        self.track_config["ransac"]["inlier_normal_angle"] = 20
        self.track_config["ransac"]["max_trans_neighbor"] = 0.02
        self.track_config["ransac"]["max_rot_deg_neighbor"] = 30
        self.track_config["ransac"]["max_trans_no_neighbor"] = 0.01
        self.track_config["ransac"]["max_rot_no_neighbor"] = 10
        self.track_config["p2p"]["max_dist"] = 0.02
        self.track_config["p2p"]["max_normal_angle"] = 45

        self.nerf_config["continual"] = True
        self.nerf_config["trunc_start"] = 0.01
        self.nerf_config["trunc"] = 0.01
        self.nerf_config["mesh_resolution"] = 0.005
        self.nerf_config["down_scale_ratio"] = 1
        self.nerf_config["fs_sdf"] = 0.1
        self.nerf_config["far"] = self.track_config["depth_processing"]["zfar"]
        self.nerf_config["datadir"] = f"{debug_dir}/nerf_with_bundletrack_online"
        self.nerf_config["notes"] = ""
        self.nerf_config["expname"] = run_name
        self.nerf_config["save_dir"] = self.nerf_config["datadir"]

        # Save config path
        self.track_cfg_path = self.output_dir / self.track_cfg_path.name
        self.nerf_cfg_path = self.output_dir / self.nerf_cfg_path.name
        yaml.dump(self.track_config, self.track_cfg_path.open("w"))
        yaml.dump(self.nerf_config, self.nerf_cfg_path.open("w"))

        self.device = device

        self.logger = get_logger("BundleSDF")

        self.load_model(
            use_nerf=use_nerf,
            run_nerf_async=run_nerf_async,
            start_nerf_keyframes=start_nerf_keyframes,
            save_results=save_results,
            use_gui=use_gui,
        )
        self.reset()

    @timer
    def load_model(
        self,
        use_nerf=True,
        run_nerf_async=True,
        start_nerf_keyframes=5,
        save_results=False,
        use_gui=False,
    ):
        self.tracker = BundleSdf(
            cfg_track_dir=str(self.track_cfg_path),
            cfg_nerf_dir=str(self.nerf_cfg_path),
            loftr_ckpt_path=self.checkpoint,
            use_nerf=use_nerf,
            run_nerf_async=run_nerf_async,
            start_nerf_keyframes=start_nerf_keyframes,
            use_gui=use_gui,
            save_results=save_results,
            logger=self.logger,
        )

        self.logger.info(f"Finish loading BundleSdf with {use_nerf=} {run_nerf_async=}")

        self.tracker_frame_idx = 0

    def reset(self):
        """Reset the pose tracker"""
        # TODO: implement reset for BundleSdf

        set_seed(0)

        # self.tracker.clear_memory()
        self.tracker_frame_idx = 0

        self.init_T_obj_cam = np.eye(4)  # Initial camera pose in object frame

    def initialize_object_pose(self, T_cam_obj: np.ndarray):
        """Initialize object pose in camera frame"""
        self.init_T_obj_cam = np.linalg.inv(T_cam_obj)

    def initialize_T_obj_cam(self, T_obj_cam: np.ndarray):
        """Initialize camera pose in object frame"""
        self.init_T_obj_cam = T_obj_cam

    @timer
    @torch.no_grad()
    def __call__(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        return_on_cpu=False,
        verbose=False,
    ) -> np.ndarray:
        """
        :param rgb_frame: Input RGB image frame, [H, W, 3] np.uint8 np.ndarray
        :param depth_frame: Input depth image frame, [H, W] np.floating np.ndarray
        :param K: camera intrinsics matrix, [3, 3] np.floating np.ndarray
        :param mask: binary target object mask, [H, W] bool np.ndarray
        :param return_on_cpu: reserved, always return as np.ndarray
        :param verbose: whether to print debug info
        :return T_obj_cam: predicted camera pose in object frame
                           [4, 4] np.float32 np.ndarray
        """
        with torch.cuda.device(self.device):
            # Process mask
            if (kernel_size := self.track_config["erode_mask"]) > 0:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask.astype(np.uint8), kernel)

            T_obj_cam = self.tracker.run(
                color=rgb_frame[:, :, ::-1],
                depth=depth_frame,
                K=K,
                id_str=f"frame_{self.tracker_frame_idx}",
                mask=mask,
                occ_mask=None,  # TODO: what is this
                pose_in_model=self.init_T_obj_cam,  # T_obj_cam
            )

            self.tracker_frame_idx += 1

        # self.tracker.on_finish()

        return T_obj_cam
