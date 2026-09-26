"""
Continuous multi-modal reward formulation (Section 4, Eq. 16-28), adapted for
low-level attitude control [roll, pitch, yaw_rate, climb_rate] instead of the
paper's idealized body-frame velocity command. Where the original reward used
the *commanded* forward/lateral velocity, this version uses the vehicle's
*actual measured* body-frame velocity (read back from AirSim kinematics after
the attitude command is executed), which better reflects real quadrotor
dynamics (banking turns bleed off speed, throttle lag, etc.).
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
    yaw_rate_norm: float,        # commanded yaw-rate action, in [-1,1]
    roll_norm: float,             # commanded roll action, in [-1,1]
    body_velocity: np.ndarray,    # measured [v_forward, v_lateral, v_z] (m/s), body frame
    cmd_smooth: np.ndarray,       # smoothed [roll,pitch,yaw_rate,climb] command
    cmd_target: np.ndarray,       # raw (pre-smoothing) [roll,pitch,yaw_rate,climb] command
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
    if o_front > config.OBS_FRONT_THRESHOLD:
        # roll is the control axis that produces sideways escape motion under
        # coordinated attitude control, so it plays the role of |v_lateral| here.
        r_obs += abs(roll_norm)

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

    total = r_progress + r_yaw + r_motion + r_obs + r_jerk + r_time + r_terminal

    info = dict(
        r_progress=r_progress,
        r_yaw=r_yaw,
        r_motion=r_motion,
        r_obs=r_obs,
        r_jerk=r_jerk,
        r_time=r_time,
        r_terminal=r_terminal,
        o_front=o_front,
        o_global=o_global,
        e_theta=e_theta,
    )
    return total, info
