"""
Continuous multi-modal reward formulation (Section 4, Eq. 16-28) over the
paper's body-frame velocity action [v_forward, v_lateral, yaw_rate]. Where
the paper's reward used the *commanded* forward/lateral velocity, this
version uses the vehicle's *actual measured* body-frame velocity (read back
from AirSim kinematics after the velocity command is executed), which better
reflects real quadrotor dynamics (inertia, drag) than the idealized
instantaneous command.
"""

import math
import numpy as np

import config
from occupancy import frontal_and_global_occupancy, avoidance_steer


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
    yaw_rate_norm: float,        # commanded yaw-rate action, in [-1,1]
    lateral_norm: float,          # commanded lateral-velocity action, in [-1,1]
    body_velocity: np.ndarray,    # measured [v_forward, v_lateral, v_z] (m/s), body frame
    cmd_smooth: np.ndarray,       # smoothed [v_forward,v_lateral,yaw_rate] command, in [-1,1]
    cmd_target: np.ndarray,       # raw (pre-smoothing) [v_forward,v_lateral,yaw_rate] command
    occupancy_grid: np.ndarray,   # 5x5 matrix G
    collided: bool,
    timed_out: bool,
):
    """
    Returns (total_reward, info_dict) following Rtotal = Rprogress + Ryaw + Rmotion
    + Robs + Rjerk + Rterminal + Rtime (Eq. 16, plus the per-step time penalty
    described in Section 4 "Step Decay and Terminal Rewards").
    """

    # --- Progress and target proximity (Eq. 17) ---
    r_progress = (d_prev - d_curr) * (config.PROGRESS_LINEAR_COEFF + config.PROGRESS_INVERSE_COEFF / (d_curr + 1.0))

    # --- Target-facing alignment (Eq. 19-20) ---
    e_theta = yaw_alignment_error_deg(current_yaw_deg, dx, dy)
    r_yaw = config.YAW_ALIGN_COEFF * (1.0 - e_theta / 180.0) - config.YAW_RATE_PENALTY_COEFF * abs(yaw_rate_norm)

    # --- Motion reward (Eq. 21), using actual measured body-frame velocity ---
    v_forward = float(body_velocity[0])
    r_motion = config.MOTION_FORWARD_COEFF * v_forward + config.MOTION_SMOOTH_COEFF * float(np.linalg.norm(body_velocity))

    # --- Obstacle-aware penalty (Eq. 22, 25-27) ---
    o_front, o_global = frontal_and_global_occupancy(occupancy_grid)
    r_obs = -config.OBS_FRONT_COEFF * o_front - config.OBS_GLOBAL_COEFF * o_global

    # --- Free-space-seeking shaping ---
    # While the frontal column is obstructed, reward lateral velocity that
    # agrees with the occupancy grid's own free-space bearing estimate (the
    # same signal the safety shield uses), so the policy is explicitly
    # taught to steer into the open part of the U-Net probability map rather
    # than only discovering that indirectly via collision penalties.
    r_freespace = 0.0
    if o_front > config.OBS_FRONT_THRESHOLD:
        v_lateral = float(body_velocity[1])
        r_obs += config.LATERAL_ESCAPE_COEFF * min(abs(v_lateral), 1.0)

        steer_lr, _danger = avoidance_steer(occupancy_grid)  # >0 => free space is to the right
        v_lateral_norm = float(np.clip(v_lateral / config.V_MAX, -1.0, 1.0))
        r_freespace = config.FREESPACE_COEFF * steer_lr * v_lateral_norm

    # --- Jerk regularization (Eq. 23-24), penalizing abrupt attitude/throttle commands ---
    r_jerk = -config.JERK_COEFF * float(np.linalg.norm(cmd_target - cmd_smooth))

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

    total = r_progress + r_yaw + r_motion + r_obs + r_freespace + r_jerk + r_time + r_terminal

    info = dict(
        r_progress=r_progress,
        r_yaw=r_yaw,
        r_motion=r_motion,
        r_obs=r_obs,
        r_freespace=r_freespace,
        r_jerk=r_jerk,
        r_time=r_time,
        r_terminal=r_terminal,
        o_front=o_front,
        o_global=o_global,
        e_theta=e_theta,
    )
    return total, info
