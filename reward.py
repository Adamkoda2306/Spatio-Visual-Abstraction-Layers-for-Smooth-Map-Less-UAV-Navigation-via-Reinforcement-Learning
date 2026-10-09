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
    milestones_hit: set,          # per-episode set of proximity-milestone keys already awarded; MUTATED in place
    shield_severity: float = 0.0,  # 0..1, how hard the safety shield intervened this step (0 if it didn't)
):
    """
    Returns (total_reward, info_dict) following Rtotal = Rprogress + Ryaw + Rmotion
    + Robs + Rjerk + Rterminal + Rtime (Eq. 16, plus the per-step time penalty
    described in Section 4 "Step Decay and Terminal Rewards").
    """

    # --- Progress and target proximity (Eq. 17) ---
    r_progress = (d_prev - d_curr) * (config.PROGRESS_LINEAR_COEFF + config.PROGRESS_INVERSE_COEFF / (d_curr + 1.0))

    # --- Distance-to-target shaping (re-added from the earlier envF.py prototype) ---
    # A small continuous penalty on raw distance plus an inverse-distance
    # bonus, on top of the Eq. 17 progress term above.
    r_distance = -config.DISTANCE_PENALTY_COEFF * d_curr + config.INVERSE_DISTANCE_COEFF / (d_curr + 1.0)

    # --- Proximity milestone bonuses, ONE-TIME per episode ---
    # The original envF.py prototype paid these every single step the UAV
    # stayed inside a distance band (e.g. +450/step within 10m), which made
    # loitering near the goal without ever entering SUCCESS_RADIUS far more
    # profitable than actually finishing -- e.g. hovering 9.5m away for 500
    # steps earned ~24,000+ reward while the episode still timed out. Each
    # milestone now pays out exactly once per episode (tracked via
    # `milestones_hit`, reset by the caller at the start of every episode),
    # so getting close still matters, but completing the approach to within
    # SUCCESS_RADIUS (R_SUCCESS) is no longer dominated by a hover reward.
    r_proximity = 0.0
    for _radius, _bonus, _key in (
        (config.MILESTONE_RADIUS_1, config.MILESTONE_BONUS_1, "m1"),
        (config.MILESTONE_RADIUS_2, config.MILESTONE_BONUS_2, "m2"),
        (config.MILESTONE_RADIUS_3, config.MILESTONE_BONUS_3, "m3"),
    ):
        if d_curr < _radius and _key not in milestones_hit:
            r_proximity += _bonus
            milestones_hit.add(_key)

    # --- Idle penalty (re-added from the earlier envF.py prototype) ---
    # Penalizes steps that make almost no progress toward or away from the
    # target, discouraging hovering/stalling.
    r_idle = -config.IDLE_PENALTY if abs(d_prev - d_curr) < config.IDLE_PROGRESS_THRESHOLD else 0.0

    # --- Target-facing alignment (Eq. 19-20) ---
    e_theta = yaw_alignment_error_deg(current_yaw_deg, dx, dy)
    r_yaw = config.YAW_ALIGN_COEFF * (1.0 - e_theta / 180.0) - config.YAW_RATE_PENALTY_COEFF * abs(yaw_rate_norm)

    # --- Motion reward (Eq. 21), using actual measured body-frame velocity ---
    v_forward = float(body_velocity[0])
    r_motion = config.MOTION_FORWARD_COEFF * v_forward + config.MOTION_SMOOTH_COEFF * float(np.linalg.norm(body_velocity))

    # --- Obstacle-aware penalty (Eq. 22, 25-27) ---
    o_front, o_global = frontal_and_global_occupancy(occupancy_grid)
    r_obs = -config.OBS_FRONT_COEFF * o_front - config.OBS_GLOBAL_COEFF * o_global

    # --- Graduated obstacle-proximity penalty ---
    # Mirrors the goal proximity milestones structurally (escalating bands),
    # but pushes the policy AWAY from danger instead of toward a target, and
    # is charged every single step the frontal occupancy stays in a band --
    # unlike the goal milestones, there is no "loiter to farm reward" risk
    # here, since this is strictly a cost, so repeating it every step is
    # exactly what's wanted: the longer the UAV stays pointed at something
    # dangerous, the more it keeps bleeding reward for it. This stacks with
    # the continuous linear r_obs term above, giving the policy a much
    # sharper, more clearly-staged signal than a single linear coefficient
    # ("this is mildly bad" -> "this is bad" -> "this is very bad") that is
    # specifically meant to be easy to learn to avoid entirely.
    r_obstacle_proximity = 0.0
    for _o_radius, _o_penalty in (
        (config.OBSTACLE_DANGER_RADIUS_1, config.OBSTACLE_DANGER_PENALTY_1),
        (config.OBSTACLE_DANGER_RADIUS_2, config.OBSTACLE_DANGER_PENALTY_2),
        (config.OBSTACLE_DANGER_RADIUS_3, config.OBSTACLE_DANGER_PENALTY_3),
    ):
        if o_front > _o_radius:
            r_obstacle_proximity -= _o_penalty

    # --- Free-space-weighted motion reward (minor addon beyond Eq. 21) ---
    # The U-Net probability map stays an obstacle detector trained purely by
    # supervised mask labels (unet.py / train_unet.py) -- it is NOT jointly
    # trained with the policy. This term simply reads that same frozen map a
    # second way: (1 - o_front) is the model's estimate of how *free* the
    # region the UAV is currently flying into is, so forward progress is
    # scaled by it -- moving fast through a region the map scores as open
    # earns close to the full motion reward, moving fast into a region it
    # scores as obstructed earns almost none (on top of the r_obs penalty
    # above). This gives the policy a direct, continuous "free space ahead
    # -> moving there is more rewarding" signal, instead of only a one-off
    # bonus when already past the obstacle threshold.
    forward_progress = max(0.0, v_forward) / config.V_MAX
    r_space = config.FREE_MOTION_COEFF * (1.0 - o_front) * min(forward_progress, 1.0)

    # --- Clear-path bonus ---
    # Extra reward, on top of r_space, specifically for making forward
    # progress while the frontal column is confidently clear (well under the
    # first danger band above) -- the direct positive counterpart to the
    # obstacle-proximity penalty: staying away from obstacles is rewarded,
    # not just approaching them punished. Tied to forward_progress (exactly
    # like r_space) so it cannot be farmed by hovering in open space.
    r_clearance = 0.0
    if o_front < config.CLEARANCE_SAFE_THRESHOLD:
        r_clearance = config.CLEARANCE_BONUS_COEFF * min(forward_progress, 1.0)

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

    # --- Safety-shield intervention penalty ---
    # airsim_env.py's hard-coded safety shield can override the policy's raw
    # action near obstacles, which means a step that "should" have collided
    # can come back with a perfectly fine r_obs (computed from the resulting
    # occupancy/velocity *after* the shield already fixed it). That breaks
    # the gradient the policy needs to learn genuine avoidance on its own --
    # it can keep flying straight at things and rely on the shield to save
    # it, since the reward rarely shows the near-miss. This term closes that
    # gap directly: it charges the policy for how hard the shield had to
    # intervene this step, independent of whether a collision actually
    # happened, so "I needed rescuing" is always a cost even when the rescue
    # worked.
    r_shield = -config.SHIELD_PENALTY_COEFF * shield_severity

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

    total = (
        r_progress + r_distance + r_proximity + r_idle
        + r_yaw + r_motion + r_obs + r_obstacle_proximity + r_space + r_clearance
        + r_freespace + r_shield + r_jerk + r_time + r_terminal
    )

    info = dict(
        r_progress=r_progress,
        r_distance=r_distance,
        r_proximity=r_proximity,
        r_idle=r_idle,
        r_yaw=r_yaw,
        r_motion=r_motion,
        r_obs=r_obs,
        r_obstacle_proximity=r_obstacle_proximity,
        r_space=r_space,
        r_clearance=r_clearance,
        r_freespace=r_freespace,
        r_shield=r_shield,
        r_jerk=r_jerk,
        r_time=r_time,
        r_terminal=r_terminal,
        o_front=o_front,
        o_global=o_global,
        e_theta=e_theta,
    )
    return total, info
