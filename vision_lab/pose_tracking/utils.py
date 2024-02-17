"""Utility functions for pose tracking"""

from typing import Tuple

import numpy as np


def depth2xyz(depth_image, intrinsics, depth_scale=1000.0) -> np.ndarray:
    """Use camera intrinsics to convert depth_image to xyz_image
    :param depth_image: [H, W] or [H, W, 1] np.uint16 np.ndarray
    :param intrinsics: [3, 3] camera intrinsics matrix
    :return xyz_image: [H, W, 3] np.float64 np.ndarray
    """
    if intrinsics.size == 4:
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    if depth_image.ndim == 3:
        assert (
            depth_image.shape[-1] == 1
        ), f"Wrong number of channels: {depth_image.shape}"
        depth_image = depth_image[..., 0]

    height, width = depth_image.shape[:2]
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_image / depth_scale
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    xyz_image = np.stack([x, y, z], axis=-1)
    return xyz_image


def interpolate_xyz(xyz_image, points) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bilinear interpolation to xyz_image
    :param xyz_image: xyz image MUST be in camera frame
                      [H, W, 3] np.float64, np.ndarray
    :param points: interpolation points in XY pixel coordinates
                   [num_points, 2] np.float32 np.ndarray
    :return points_xyz: interpolated xyz values at points
                        [num_points, 3] np.float64 np.ndarray
    :return depth_valid_mask: points valid mask with positive depth values
                              [num_points,] bool np.ndarray
    """
    x, y = points.T
    x_low = np.floor(x).astype(int)
    y_low = np.floor(y).astype(int)
    x_high = np.ceil(x).astype(int)
    y_high = np.ceil(y).astype(int)
    xyz_ll = xyz_image[y_low, x_low]
    xyz_lh = xyz_image[y_low, x_high]
    xyz_hl = xyz_image[y_high, x_low]
    xyz_hh = xyz_image[y_high, x_high]
    x, y, x_low, x_high, y_low, y_high = map(
        lambda v: v[:, None], [x, y, x_low, x_high, y_low, y_high]
    )

    points_xyz = (
        xyz_ll * (y_high - y) * (x_high - x)
        + xyz_lh * (y_high - y) * (x - x_low)
        + xyz_hl * (y - y_low) * (x_high - x)
        + xyz_hh * (y - y_low) * (x - x_low)
    )

    # Depth valid means all 4 neighbours have valid depths
    depth_valid_mask = (
        (xyz_ll[:, 2] > 0)
        & (xyz_lh[:, 2] > 0)
        & (xyz_hl[:, 2] > 0)
        & (xyz_hh[:, 2] > 0)
    )

    return points_xyz, depth_valid_mask


def umeyama(src_pts, dst_pts, with_scaling=False) -> np.ndarray:
    """Estimate a (similarity) transfromation matrix that minimizes the MSE

    np.square(src_pts @ T_dst_src[:3, :3].T + T_dst_src[:3, 3] - dst_pts).sum() / N

    :param src_pts: [N, 3] np.floating np.ndarray
    :param dst_pts: [N, 3] np.floating np.ndarray
    :param with_scaling: estimate a similarity transformation (rigid + unit scaling)
    :return T_dst_src: [4, 4] transformation matrix that maps src_pts to dst_pts
    """
    one_over_num_pts = 1.0 / src_pts.shape[0]

    src_pts_mean = src_pts.sum(axis=0) * one_over_num_pts
    dst_pts_mean = dst_pts.sum(axis=0) * one_over_num_pts

    src_pts_centered = src_pts - src_pts_mean
    dst_pts_centered = dst_pts - dst_pts_mean

    cov_matrix = dst_pts_centered.T @ src_pts_centered * one_over_num_pts
    U, S, Vh = np.linalg.svd(cov_matrix)

    if np.linalg.det(U @ Vh) < 0:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]

    R = U @ Vh

    scale = 1
    if with_scaling:
        src_pts_var = np.square(src_pts_centered).sum() * one_over_num_pts
        scale = S.sum() / src_pts_var

    T_dst_src = np.eye(4)
    T_dst_src[:3, :3] = sR = scale * R
    T_dst_src[:3, 3] = dst_pts_mean - sR @ src_pts_mean
    return T_dst_src
