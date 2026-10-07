"""
Central configuration for the spatio-visual map-less UAV navigation framework.
Values are taken directly from the paper (Sections 3-4, Table 2).
"""

import os

# --------------------------------------------------------------------------
# Image / perception
# --------------------------------------------------------------------------
IMG_SIZE = 256                 # AirSim FPV camera resolution (256x256x3 RGB)
OCC_GRID = 5                   # compressed occupancy matrix size (5x5)
UNET_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "unet.pt")
CAMERA_HFOV_DEG = 90.0          # front camera horizontal FOV (settings.json CaptureSettings.FOV_Degrees)

# --------------------------------------------------------------------------
# Live occupancy-map HUD (live_view.py) -- purely a visualization, never read
# by the policy or reward. Shows the 5x5 probability grid with markers for
# the goal bearing and the currently executed steering direction.
# --------------------------------------------------------------------------
LIVE_VIEW = True
LIVE_VIEW_CELL_PX = 70          # each grid cell is rendered this many pixels square

# --------------------------------------------------------------------------
# State / action space
# --------------------------------------------------------------------------
STATE_DIM = OCC_GRID * OCC_GRID + 3   # 25 (flattened grid) + dx, dy, dz = 28

# Action space (Eq. 9): continuous body-frame [v_forward, v_lateral, yaw_rate]
# in [-1, 1], driven through AirSim's moveByVelocityBodyFrameAsync (Section
# 3.4). Altitude is held automatically via the Eq. 13 cruise-altitude
# controller, not commanded by the policy.
ACTION_DIM = 3

# --------------------------------------------------------------------------
# Physical UAV motion constraints (Eq. 10-13)
# --------------------------------------------------------------------------
V_MAX = 5.0            # m/s, forward/lateral velocity scale
OMEGA_MAX = 5.0         # deg/s, yaw rate scale
H_MAX = 10.0            # m, cruise altitude (AirSim NED: z = -H_MAX)

# --------------------------------------------------------------------------
# Velocity smoothing / jerk regularization
# --------------------------------------------------------------------------
ALPHA_SMOOTH = 0.01   # first-order low-pass EMA responsiveness coefficient
                        # (Table 2: alpha in {0.001, 0.005, 0.01, 0.05, 0.1};
                        # 0.005-0.01 gave the best reward / episode length trade-off)

# --------------------------------------------------------------------------
# Reward shaping constants (Section 4)
# --------------------------------------------------------------------------
R_TIME = -0.02
R_SUCCESS = 2000.0
R_COLLISION = -1000.0
R_TIMEOUT = -1000.0
SUCCESS_RADIUS = 1.0          # meters
MAX_EPISODE_STEPS = 500        # shorter episodes -> more terminal-reward samples per training budget
OBS_FRONT_THRESHOLD = 0.3
JERK_COEFF = 0.05
YAW_ALIGN_COEFF = 0.2
LATERAL_ESCAPE_COEFF = 1.0
YAW_RATE_PENALTY_COEFF = 0.4
MOTION_FORWARD_COEFF = 0.3
MOTION_SMOOTH_COEFF = 0.4
OBS_FRONT_COEFF = 4.0
OBS_GLOBAL_COEFF = 1.0
PROGRESS_LINEAR_COEFF = 20.0
PROGRESS_INVERSE_COEFF = 100.0

# Free-space-weighted motion reward: forward progress is scaled by
# (1 - frontal occupancy), i.e. the same frozen U-Net probability map is
# additionally read as "how rewarding is moving in this direction" --
# moving through cells the map scores as open earns close to the full
# motion reward, moving into cells it scores as obstructed earns almost
# none. The U-Net itself is untouched (still a supervised, fixed-during-RL
# obstacle detector, see unet.py); only the reward shaping is extended.
FREE_MOTION_COEFF = 0.4

# Free-space-seeking shaping: while something blocks the frontal column,
# reward lateral velocity that agrees with the occupancy-grid's estimated
# free-space bearing (see occupancy.avoidance_steer), so the policy is
# explicitly taught "steer toward the open part of the probability map"
# instead of relying only on collision penalties to discover that.
FREESPACE_COEFF = 0.5

# --------------------------------------------------------------------------
# Perception-driven safety shield (airsim_env.py only -- does NOT touch the
# reward function above). The reward still scores the agent on its own raw
# action exactly as before; this just clamps what actually gets flown so the
# U-Net occupancy grid can veto/soften a command that's about to fly straight
# into an obstacle, instead of relying solely on the reward gradient to teach
# that behavior over millions of steps.
# --------------------------------------------------------------------------
AVOID_DANGER_THRESHOLD = 0.15   # eye-level occupancy (0-1) above which the shield starts engaging
AVOID_DANGER_HARD = 0.55        # above this, forward velocity is cut almost to zero regardless of the curve below
AVOID_STEER_GAIN = 1.5          # added lateral-velocity correction (toward free space) at full danger severity
AVOID_BRAKE_GAIN = 0.9          # fraction of forward velocity cut at full danger severity
AVOID_STEER_SIGN = 1.0          # flip to -1.0 if the drone is observed steering toward obstacles instead of away
AVOID_EMA_ALPHA = 0.35          # smooths the per-frame danger/steer signal before it drives the shield, so
                                # single-frame U-Net noise doesn't jerk the lateral/forward output step to step
                                # (this is what was making the drone visibly shake in 3rd-person view)

# --------------------------------------------------------------------------
# AirSim connection
# --------------------------------------------------------------------------
AIRSIM_IP = "127.0.0.1"
VEHICLE_NAME = ""              # default vehicle in AirSimNH settings.json
CAMERA_NAME = "0"              # front FPV camera id
# Seconds each velocity command is held before re-planning. AirSim's
# moveByVelocityBodyFrameAsync holds the commanded velocity for exactly this
# long; the Python step loop also does an image capture + U-Net forward pass
# + collision check in between commands, which takes real wall-clock time on
# top of this duration. Lower it only if your machine's per-step loop is
# fast enough that flight still looks smooth.
ACTION_DURATION = 0.4

# --------------------------------------------------------------------------
# Target / episode sampling (AirSimNH suburban block, NED frame, meters)
# --------------------------------------------------------------------------
START_POSITION = (0.0, 0.0, -H_MAX)
GOAL_SAMPLING_RADIUS_MIN = 40.0
GOAL_SAMPLING_RADIUS_MAX = 110.0

# Fixed deployment goal, e.g. (25.0, -40.0, -15.0) in NED meters (z negative = up).
# Leave as None during training to keep sampling random goals for generalization;
# set it (or pass --goal_x/--goal_y/--goal_z / reset(options={"goal": (x,y,z)}))
# to fly to one specific destination.
FIXED_GOAL = None

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TOTAL_TIMESTEPS = 1_000_000
