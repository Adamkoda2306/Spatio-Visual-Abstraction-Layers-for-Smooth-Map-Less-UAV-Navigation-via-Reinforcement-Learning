"""
Bounded, continuous reward over the paper's body-frame velocity action
[v_forward, v_lateral, yaw_rate]:

    R = r_progress + r_obs + r_yaw + r_jerk + r_time + r_terminal

Every per-step term is bounded (r_progress in [-1, 1], r_obs in [-2, 0],
r_yaw in [-0.1, 0], r_jerk and r_time tiny), and the terminal payouts are
+100 / -60 / -60, so an episode return stays in a range a value head can
actually fit. Obstacle avoidance is learned by the policy itself from this
signal -- there is no heuristic shield term and no one-off milestone payouts.
"""

import math
import numpy as np

import config
from occupancy import frontal_and_global_occupancy


def yaw_alignment_error_deg(current_yaw_deg: float, dx: float, dy: float) -> float:
    """
    Eq. 19: e_theta = |(theta_target - theta_yaw + 180) mod 360 - 180|
    theta_target is the bearing from the UAV to the goal, in the horizontal plane.
    AirSim NED: x=North, y=East, so bearing = atan2(dy, dx).
    """
    theta_target = math.degrees(math.atan2(dy, dx))
    diff = (theta_target - current_yaw_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


def compute_reward(
    d_prev: float,
    d_curr: float,
    current_yaw_deg: float,
    dx: float,
    dy: float,
    action: np.ndarray,           # this step's raw [v_forward,v_lateral,yaw_rate] action, in [-1,1]
    prev_action: np.ndarray,      # previous step's raw action (zeros on the first step)
    occupancy_grid: np.ndarray,   # 5x5 matrix G (logged only, not rewarded)
    collided: bool,
    timed_out: bool,
    dt: float = config.ACTION_DURATION,  # measured seconds since the previous sensor read
    lidar_front_m: float = config.LIDAR_MAX_RANGE,  # min LiDAR range (meters) in the frontal sectors this step
    lidar_global_m: float = config.LIDAR_MAX_RANGE,  # min LiDAR range (meters) over all sectors this step
):
    """Returns (total_reward, info_dict)."""

    # --- Progress toward the goal ---
    # Distance closed this step, normalized by the farthest the UAV can move
    # in the elapsed step time (V_MAX * dt), so it lies in [-1, 1]. dt is the
    # measured time between sensor reads, since real-time steps vary in length.
    max_step = config.V_MAX * max(dt, 1e-3)
    r_progress = float(np.clip((d_prev - d_curr) / max_step, -1.0, 1.0))

    # --- Smooth LiDAR obstacle barrier ---
    # 0 beyond LIDAR_SAFE_DIST, growing linearly to -LIDAR_PENALTY_COEFF as the
    # nearest frontal return approaches 0 m (side/rear sectors at half weight,
    # since the drone also strafes): a continuous gradient away from obstacles
    # rather than stacked step penalties.
    barrier_front = max(0.0, 1.0 - lidar_front_m / config.LIDAR_SAFE_DIST)
    barrier_any = max(0.0, 1.0 - lidar_global_m / config.LIDAR_SAFE_DIST)
    r_obs = -config.LIDAR_PENALTY_COEFF * max(barrier_front, config.LIDAR_SIDE_WEIGHT * barrier_any)

    # --- Target-facing alignment (Eq. 19-20) ---
    e_theta = yaw_alignment_error_deg(current_yaw_deg, dx, dy)
    r_yaw = -config.YAW_ALIGN_COEFF * e_theta / 180.0

    # --- Action-change (jerk) penalty (Eq. 23-24) ---
    # Cost on how much the raw action changed since the last step, so the
    # policy itself learns smooth outputs instead of relying on the EMA filter.
    r_jerk = -config.JERK_COEFF * float(np.linalg.norm(action - prev_action))

    # --- Step decay ---
    r_time = config.R_TIME

    # --- Terminal constraints (Eq. 28) ---
    r_terminal = 0.0
    if d_curr < config.SUCCESS_RADIUS:
        r_terminal = config.R_SUCCESS
    elif collided:
        r_terminal = config.R_COLLISION
    elif timed_out:
        r_terminal = config.R_TIMEOUT

    total = r_progress + r_obs + r_yaw + r_jerk + r_time + r_terminal

    o_front, o_global = frontal_and_global_occupancy(occupancy_grid)
    info = dict(
        r_progress=r_progress,
        r_obs=r_obs,
        r_yaw=r_yaw,
        r_jerk=r_jerk,
        r_time=r_time,
        r_terminal=r_terminal,
        o_front=o_front,
        o_global=o_global,
        e_theta=e_theta,
    )
    return total, info
