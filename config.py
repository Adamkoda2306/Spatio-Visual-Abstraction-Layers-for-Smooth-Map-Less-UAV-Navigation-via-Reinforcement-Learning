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

# --------------------------------------------------------------------------
# State / action space
# --------------------------------------------------------------------------
STATE_DIM = OCC_GRID * OCC_GRID + 3   # 25 (flattened grid) + dx, dy, dz = 28

# Action space extended beyond the paper's [v_forward, v_lateral, yaw_rate]
# body-frame velocity command to full low-level attitude control
# [roll, pitch, yaw_rate, climb_rate] in [-1, 1], driven through AirSim's
# moveByRollPitchYawrateZAsync — this produces coordinated banking turns and
# smooth throttle changes like a real flight controller, instead of an
# idealized instantaneous velocity vector.
ACTION_DIM = 4

# --------------------------------------------------------------------------
# Physical UAV motion constraints
# --------------------------------------------------------------------------
V_MAX = 5.0            # m/s, reserved for reward normalization
OMEGA_MAX = 5.0         # deg/s, yaw rate scale
H_MAX = 10.0            # m, default target altitude (AirSim NED: z = -H_MAX)

MAX_ROLL_DEG = 15.0     # deg, max commanded bank angle
MAX_PITCH_DEG = 15.0    # deg, max commanded pitch angle
MAX_CLIMB_RATE = 2.0    # m/s, max commanded vertical speed (throttle axis)
ALT_MIN = 2.0           # m, closest allowed altitude to the ground
ALT_MAX = 60.0          # m, highest allowed altitude

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
R_SUCCESS = 500.0
R_COLLISION = -250.0
R_TIMEOUT = -250.0
SUCCESS_RADIUS = 1.0          # meters
MAX_EPISODE_STEPS = 1000
OBS_FRONT_THRESHOLD = 0.3
JERK_COEFF = 0.05
YAW_ALIGN_COEFF = 0.10
LATERAL_ESCAPE_COEFF = 0.30
YAW_RATE_PENALTY_COEFF = 0.05
MOTION_FORWARD_COEFF = 0.03
MOTION_SMOOTH_COEFF = 0.01
OBS_FRONT_COEFF = 6.0
OBS_GLOBAL_COEFF = 1.0
PROGRESS_LINEAR_COEFF = 30.0
PROGRESS_INVERSE_COEFF = 20.0

# --------------------------------------------------------------------------
# Perception-driven safety shield (airsim_env.py only -- does NOT touch the
# reward function above). The reward still scores the agent on its own raw
# action exactly as before; this just clamps what actually gets flown so the
# U-Net occupancy grid can veto/soften a command that's about to fly straight
# into an obstacle, instead of relying solely on the reward gradient to teach
# that behavior over millions of steps.
# --------------------------------------------------------------------------
AVOID_DANGER_THRESHOLD = 0.15   # eye-level occupancy (0-1) above which the shield starts engaging
AVOID_DANGER_HARD = 0.55        # above this, pitch is cut almost to zero regardless of the curve below
AVOID_STEER_GAIN = 2.0          # added lateral (roll) correction at full danger severity
AVOID_BRAKE_GAIN = 0.9          # fraction of forward pitch cut at full danger severity
AVOID_STEER_SIGN = 1.0          # flip to -1.0 if the drone is observed steering toward obstacles instead of away
AVOID_EMA_ALPHA = 0.35          # smooths the per-frame danger/steer signal before it drives the shield, so
                                # single-frame U-Net noise doesn't jerk the roll/pitch output step to step
                                # (this is what was making the drone visibly shake in 3rd-person view)

# --------------------------------------------------------------------------
# AirSim connection
# --------------------------------------------------------------------------
AIRSIM_IP = "127.0.0.1"
VEHICLE_NAME = ""              # default vehicle in AirSimNH settings.json
CAMERA_NAME = "0"              # front FPV camera id
# Seconds each attitude command is held before re-planning. AirSim's
# moveByRollPitchYawrateZAsync holds the commanded attitude for exactly this
# long, then (if nothing else blocking it) effectively settles; the Python
# step loop also does an image capture + U-Net forward pass + collision
# check in between commands, which takes real wall-clock time on top of
# this duration. If that per-step overhead is a large fraction of
# ACTION_DURATION, the drone visibly pulses between "banking" and
# "leveling out" every step -- the shaking seen in 3rd-person view. 0.4s
# gives that overhead more headroom relative to the hold time than the
# original 0.2s; lower it only if your machine's per-step loop is fast
# enough that flight still looks smooth.
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
