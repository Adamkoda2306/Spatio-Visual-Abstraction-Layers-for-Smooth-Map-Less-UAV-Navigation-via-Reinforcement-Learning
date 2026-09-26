"""
Spatial compression of the dense U-Net obstacle probability map into a
lightweight 5x5 occupancy matrix (Section 3.3, Eq. 4-5) and the quadrant-based
obstacle-mitigation features used by the reward function (Eq. 25).
"""

import numpy as np
import cv2

import config


def pool_to_grid(prob_map: np.ndarray, grid_size: int = config.OCC_GRID) -> np.ndarray:
    """
    prob_map:  (256,256) float32 obstacle probability field in [0,1]
    returns:   (grid_size, grid_size) compressed occupancy matrix G
    Uses cv2.INTER_AREA (Spatial Area Interpolation pooling, Eq. 4).
    """
    grid = cv2.resize(prob_map.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return grid


def flatten_grid(grid: np.ndarray) -> np.ndarray:
    """Eq. 5: G -> V_visual (flattened 25-dim vector)."""
    return grid.flatten()


def frontal_and_global_occupancy(grid: np.ndarray):
    """
    Eq. 25: frontal safety column = rows 0-1, cols 2-3 of the 5x5 grid.
    O_front = mean of that sub-quadrant, O_global = mean of the full grid.
    """
    front_block = grid[0:2, 2:4]
    o_front = float(np.mean(front_block))
    o_global = float(np.mean(grid))
    return o_front, o_global
