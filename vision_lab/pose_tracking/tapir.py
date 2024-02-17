import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..point_tracking import TAPIR
from ..point_tracking.utils import sample_points_from_mask
from ..utils.timer import timer
from .utils import depth2xyz, interpolate_xyz, umeyama


class TAPIRPoseTracker:
    """TAPIR for object pose tracking"""

    def __init__(
        self,
        root_path: Union[str, Path] = "/rl_benchmark/tapnet",
        model_variant="causal",
        resize_shape=(256, 256),
        num_points=64,
        reinit_point_every=8,
        # run_name="debug_run",
        verbose_level=0,
        device="cuda",
    ):
        """
        :param resize_shape: Resize input frames to this size for TAPIR, [H, W]
        :param num_points: Number of tracking points for TAPIR
        :param reinit_point_every: Reinit TAPIR tracking points every K frames
        :param run_name: directory name under root_path for saving debug info.
        :param verbose_level: logging verbose level for BundleTrack (max=2)
        """
        root_path = Path(root_path)
        self.point_tracker = TAPIR(
            root_path, model_variant, resize_shape, num_points, device=device
        )

        self.checkpoint = self.point_tracker.checkpoint
        self.model_variant = self.point_tracker.model_variant

        self.resize_shape = self.point_tracker.resize_shape
        self.num_points = self.point_tracker.num_points
        self.reinit_point_every = reinit_point_every

        # # Create output directory
        # self.output_dir = root_path / "logs" / run_name
        # shutil.rmtree(self.output_dir, ignore_errors=True)
        # self.output_dir.mkdir(parents=True)

        self.device = device

        self.reset()

    def reset(self):
        """Reset the pose tracker"""
        # TODO: implement reset for TAPIRPoseTracker
        self.point_tracker.reset()
        self.tracker_frame_idx = 0

        self.prev_pts_xyz = None  # [num_points, 3] xyz in camera frame
        self.prev_pts_mask = None  # [num_points,] previous points valid mask
        self.prev_T_cam_obj = np.eye(4)  # Initial object pose in camera frame

    def initialize_object_pose(self, T_cam_obj: np.ndarray):
        """Initialize object pose in camera frame"""
        self.prev_T_cam_obj = T_cam_obj

    def initialize_T_obj_cam(self, T_obj_cam: np.ndarray):
        """Initialize camera pose in object frame"""
        self.prev_T_cam_obj = np.linalg.inv(T_obj_cam)

    @staticmethod
    def _compute_umeyama_transform(src_pts, dst_pts, inlier_ratio=0.8) -> np.ndarray:
        """Compute a rigid transformation using Umeyama algorithm
        :param src_pts: [N, 3] np.floating np.ndarray
        :param dst_pts: [N, 3] np.floating np.ndarray
        :param inlier_ratio: assumed inlier ratio, used for additional refinement
        :return T_dst_src: [4, 4] transformation matrix that maps src_pts to dst_pts
        """
        T_dst_src = umeyama(src_pts, dst_pts)
        diff = dst_pts - (src_pts @ T_dst_src[:3, :3].T + T_dst_src[:3, 3])
        error = np.square(diff).sum(axis=1)
        inlier_cnt = int(len(error) * inlier_ratio)
        if inlier_cnt <= 3:
            return T_dst_src
        inlier_idx = error.argsort()[:inlier_cnt]
        return umeyama(src_pts[inlier_idx], dst_pts[inlier_idx])

    @timer
    def __call__(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        K: np.ndarray,
        mask: Optional[np.ndarray] = None,
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
                           [4, 4] np.float64 np.ndarray
        """
        # Convert depth frame
        xyz_image = depth2xyz(depth_frame, K, depth_scale=1.0)

        points_px = None
        do_reinit_point = self.tracker_frame_idx % self.reinit_point_every == 0
        if do_reinit_point:
            assert mask is not None, "Reinit with no mask"
            # If new points are selected, previous points need to also
            #   be tracked to calculate pose
            if self.tracker_frame_idx > 0:
                points_px, visible = self.point_tracker(rgb_frame)
                cur_pts_xyz, depth_valid_mask = interpolate_xyz(xyz_image, points_px)
                # point needs to be on object mask AND visible AND has valid depth
                cur_pts_mask = (
                    mask[points_px[:, 1].astype(int), points_px[:, 0].astype(int)]
                    & visible
                    & depth_valid_mask
                )
                # Umeyama points should be valid in both current and previous frames
                umeyama_mask = cur_pts_mask & self.prev_pts_mask
                T_cur_prev = self._compute_umeyama_transform(
                    self.prev_pts_xyz[umeyama_mask], cur_pts_xyz[umeyama_mask]
                )
                cur_T_cam_obj = T_cur_prev @ self.prev_T_cam_obj

            # Reinit points
            points_px = sample_points_from_mask(mask, self.num_points)

        points_px, visible = self.point_tracker(rgb_frame, points_px)
        cur_pts_xyz, depth_valid_mask = interpolate_xyz(xyz_image, points_px)

        if mask is None:
            # point needs to be visible AND has valid depth
            cur_pts_mask = visible & depth_valid_mask
        else:
            # point needs to be on object mask AND visible AND has valid depth
            cur_pts_mask = (
                mask[points_px[:, 1].astype(int), points_px[:, 0].astype(int)]
                & visible
                & depth_valid_mask
            )

        if self.tracker_frame_idx == 0:
            cur_T_cam_obj = self.prev_T_cam_obj  # initial object pose in camera frame
        elif not do_reinit_point:
            # Umeyama points should be valid in both current and previous frames
            umeyama_mask = cur_pts_mask & self.prev_pts_mask
            T_cur_prev = self._compute_umeyama_transform(
                self.prev_pts_xyz[umeyama_mask], cur_pts_xyz[umeyama_mask]
            )
            cur_T_cam_obj = T_cur_prev @ self.prev_T_cam_obj

        self.prev_pts_xyz = cur_pts_xyz  # [num_points, 3] xyz in camera frame
        self.prev_pts_mask = cur_pts_mask  # [num_points,] points valid mask
        self.prev_T_cam_obj = cur_T_cam_obj  # Object pose in camera frame
        self.tracker_frame_idx += 1

        return np.linalg.inv(cur_T_cam_obj)  # T_obj_cam
