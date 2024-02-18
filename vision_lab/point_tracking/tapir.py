import functools
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tapnet import tapir_model
from tapnet.utils import transforms

from ..utils.timer import timer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # disable JAX GPU preallocation


""" Build TAPIR model
# Internally, the tapir model has three stages of processing: computing
# image features (get_feature_grids), extracting features for each query point
# (get_query_features), and estimating trajectories given query features and
# the feature grids where we want to track (estimate_trajectories).  For
# tracking online, we need extract query features on the first frame only, and
# then call estimate_trajectories on one frame at a time.
"""


def build_online_model_init(frames, query_points):
    """Initialize query features for the query points."""
    model = tapir_model.TAPIR(
        use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
    )
    feature_grids = model.get_feature_grids(frames, is_training=False)
    query_features = model.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    return query_features


def build_online_model_predict(frames, query_features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(
        use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
    )
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories["causal_context"]
    del trajectories["causal_context"]
    return {k: v[-1] for k, v in trajectories.items()}, causal_context


""" Utility functions """


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
        occlusions: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
        visibles: [num_points, num_frames], bool
    """
    pred_occ = jax.nn.sigmoid(occlusions)
    pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
    visibles = pred_occ < 0.5  # threshold
    return visibles


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def construct_initial_causal_state(num_points, num_resolutions):
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()}
    return [fake_ret] * num_resolutions * 4


def convert_select_points_to_query_points(frame, points):
    """Convert select points to query points.

    Args:
        points: [num_points, 2], [x, y]
    Returns:
        query_points: [num_points, 3], [t, y, x]
    """
    points = np.stack(points)
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points


class TAPIR:
    """TAPIR for point tracking"""

    CHECKPOINTS = {
        "causal": "checkpoints/causal_tapir_checkpoint.npy",
    }

    def __init__(
        self,
        model_variant="causal",
        resize_shape=(256, 256),
        num_points=64,
        device="cuda",
    ):
        """
        :param resize_shape: Resize input frames to this size, [H, W]
        :param num_points: Number of tracking points
        """
        raise NotImplementedError(
            "Download checkpoint locally: prev in /rl_benchmark/tapnet/checkpoints"
        )

        self.checkpoint = root_path / self.CHECKPOINTS[model_variant]
        self.model_variant = model_variant

        self.resize_shape = resize_shape
        self.num_points = num_points

        self.device = device

        self.load_model()
        self.reset()

    @timer
    def load_model(self):
        ckpt_state = np.load(self.checkpoint, allow_pickle=True).item()
        params, state = ckpt_state["params"], ckpt_state["state"]

        online_init = hk.transform_with_state(build_online_model_init)
        online_init_apply = jax.jit(online_init.apply)

        online_predict = hk.transform_with_state(build_online_model_predict)
        online_predict_apply = jax.jit(online_predict.apply)

        rng = jax.random.PRNGKey(42)
        self.online_init_apply = functools.partial(
            online_init_apply, params=params, state=state, rng=rng
        )
        self.online_predict_apply = functools.partial(
            online_predict_apply, params=params, state=state, rng=rng
        )

        # ----- JIT compile with dummy inputs ----- #
        points = np.stack(
            [
                np.random.randint(0, 848, self.num_points),
                np.random.randint(0, 480, self.num_points),
            ],
            axis=1,
        )
        query_points = convert_select_points_to_query_points(0, points)
        query_points = transforms.convert_grid_coordinates(
            query_points,
            (1, 480, 848),
            (1, *self.resize_shape),
            coordinate_format="tyx",
        )
        frame = np.random.randint(0, 256, self.resize_shape + (3,), dtype=np.uint8)

        # Initialize first frame features & states for tapir point tracking
        query_features, _ = self.online_init_apply(
            frames=preprocess_frames(frame[None, None]), query_points=query_points[None]
        )
        causal_state = construct_initial_causal_state(
            query_points.shape[0] * 2, len(query_features.resolutions) - 1
        )

        # Tracking
        for _ in range(2):
            (prediction, causal_state), _ = self.online_predict_apply(
                frames=preprocess_frames(frame[None, None]),
                query_features=query_features,
                causal_context=causal_state,
            )
            track = prediction["tracks"][0]  # [N, 1, 2], xy coords
            occlusion = prediction["occlusion"][0]  # [N, 1]
            expected_dist = prediction["expected_dist"][0]  # [N, 1]
            visible = postprocess_occlusions(occlusion, expected_dist)  # (N, 1), bool

        self.tracker_frame_idx = 0

    def reset(self):
        """Reset the point tracker"""
        # TODO: implement reset for TAPIR

        np.random.seed(42)

        # self.tracker.clear_memory()

        self.query_features = None
        self.causal_state = None
        self.tracker_frame_idx = 0

    @timer
    def __call__(
        self,
        frame: np.ndarray,
        points: Optional[np.ndarray] = None,
        return_on_cpu=True,
        verbose=False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[jax.Array, jax.Array]]:
        """
        :param frame: Input RGB image frame, [H, W, 3] np.uint8 np.ndarray
        :param points: points in XY pixel coordinates
                       [num_points, 2] np.int64 np.ndarray
        :param return_on_cpu: whether to return masks as jax.Array or np.ndarray
        :param verbose: whether to print debug info
        :return points: tracked points in XY pixel coordinates
                        [num_points, 2] np.float32 np.ndarray
        :return visible: tracked points visibility prediction
                         [num_points,] bool np.ndarray
        """
        # Process frame
        frame_shape = frame.shape[:2]
        frame = cv2.resize(frame, self.resize_shape[::-1])
        frame = preprocess_frames(frame[None, None])

        if points is not None:  # reinit with these points
            query_points = convert_select_points_to_query_points(0, points)  # tyx
            query_points = transforms.convert_grid_coordinates(
                query_points,
                (1, *frame_shape),
                (1, *self.resize_shape),
                coordinate_format="tyx",
            )

            # Initialize first frame features & states for tapir point tracking
            self.query_features, _ = self.online_init_apply(
                frames=frame, query_points=query_points[None]
            )
            self.causal_state = construct_initial_causal_state(
                query_points.shape[0] * 2, len(self.query_features.resolutions) - 1
            )

        (prediction, self.causal_state), _ = self.online_predict_apply(
            frames=frame,
            query_features=self.query_features,
            causal_context=self.causal_state,
        )
        tracked_points = prediction["tracks"][0]  # [N, 1, 2], xy coords
        occlusion = prediction["occlusion"][0]  # [N, 1]
        expected_dist = prediction["expected_dist"][0]  # [N, 1]
        visible = postprocess_occlusions(occlusion, expected_dist)  # (N, 1), bool

        tracked_points = transforms.convert_grid_coordinates(
            tracked_points, self.resize_shape[::-1], frame_shape[::-1]
        )[:, 0, :]  # (num_points, 2)

        self.tracker_frame_idx += 1

        if return_on_cpu:
            return np.asarray(tracked_points), np.asarray(visible)[..., 0]  # gpu to cpu
        else:
            return tracked_points, visible[..., 0]
