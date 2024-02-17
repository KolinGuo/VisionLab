"""Utility functions for point tracking"""

import numpy as np
from scipy import ndimage


def sample_points_from_mask(mask: np.ndarray, num_points: int) -> np.ndarray:
    """Sample random points from mask
    :param mask: [H, W] bool np.ndarray
    :param num_points: number of points to sample
    :return points: points in XY pixel coordinates
                    [num_points, 2] np.int64 np.ndarray
    """
    assert mask.dtype == bool, f"{mask.dtype = }"
    # remove boundary points, but can affect rotation accuracy
    mask_filtered = ndimage.minimum_filter(mask, size=20)

    yy, xx = np.nonzero(mask_filtered)

    # randomly select num_track_points points for new tracking
    idx = np.random.choice(yy.size, num_points, replace=False)
    points = np.stack([xx[idx], yy[idx]], axis=1)
    return points
