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


_COL_POS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # left -> right across the 5 grid columns


def avoidance_steer(grid: np.ndarray):
    """
    Perception-driven obstacle-avoidance signal derived straight from the
    U-Net occupancy grid, independent of the reward function. Used by
    airsim_env.py as a safety shield on the *executed* control command
    (not on the reward), so it never changes the learning signal.

    Returns:
        steer_lr: in [-1, 1]. Positive = steer right (obstacle mass is
                  weighted toward the left), negative = steer left.
        danger:   in [0, 1]. How urgently something is blocking the path
                  directly ahead.

    `danger` is biased toward the eye-level center block (rows 1-3, the
    vertical middle of the frame, cols 1-3) and blends mean with max so a
    compact obstacle (a pole, a tree trunk) that doesn't fill the whole
    vertical field of view isn't diluted away by clear sky above / ground
    below it in the same columns -- a plain full-grid mean badly
    underestimates exactly that case, which is what let the drone get close
    enough to still collide even with the shield engaged.
    """
    center = grid[1:4, 1:4]
    danger = 0.5 * float(center.mean()) + 0.5 * float(center.max())

    eye_level_cols = grid[1:4, :].mean(axis=0)  # shape (5,), left -> right, eye-level band only
    weight = float(eye_level_cols.sum()) + 1e-6
    obstacle_bearing = float((eye_level_cols * _COL_POS).sum() / weight)  # -1 (left) .. +1 (right)
    steer_lr = -obstacle_bearing

    return steer_lr, danger
