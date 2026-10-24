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
# LiDAR sensor (settings.json "LidarSensor1") -- real metric range data,
# complementing the camera/U-Net occupancy map. The camera stays
# responsible for *where* free space is (wide 90 deg FOV, semantic
# understanding of roads/vegetation/buildings via the trained U-Net);
# LiDAR is responsible for *how far away* the nearest obstacle actually is,
# straight from ray returns -- far more reliable for deciding how urgently
# to brake/steer than a probability heatmap alone. See lidar.py.
# --------------------------------------------------------------------------
LIDAR_NAME = "LidarSensor1"
LIDAR_NUM_SECTORS = 7            # horizontal angular bins across the sensor's FOV, left -> right
LIDAR_HFOV_DEG = 180.0           # matches settings.json HorizontalFOVStart/End (-90..90)
LIDAR_MAX_RANGE = 20.0           # meters; no return (or a return beyond this) counts as fully clear
LIDAR_STEER_WEIGHT = 0.6         # blend weight for lidar vs camera steering signal in the safety shield (0..1)

# --------------------------------------------------------------------------
# Live occupancy-map HUD (live_view.py) -- purely a visualization, never read
# by the policy or reward. Shows the 5x5 probability grid with markers for
# the goal bearing and the currently executed steering direction.
# --------------------------------------------------------------------------
LIVE_VIEW = False
LIVE_VIEW_CELL_PX = 70          # each grid cell is rendered this many pixels square

# --------------------------------------------------------------------------
# State / action space
# --------------------------------------------------------------------------
# Action space (Eq. 9): continuous body-frame [v_forward, v_lateral, yaw_rate]
# in [-1, 1], driven through AirSim's moveByVelocityBodyFrameAsync (Section
# 3.4). Altitude is held automatically via the Eq. 13 cruise-altitude
# controller, not commanded by the policy.
ACTION_DIM = 3

# State (Eq. 7, extended):
#   25  flattened camera/U-Net occupancy grid  (Vvisual, Eq. 5)
#    3  relative target displacement [dx, dy, dz]  (Eq. 6), divided by
#       GOAL_OBS_SCALE and clipped to [-1, 1]
#    1  goal bearing relative to heading, in [-1, 1] (0 = dead ahead,
#       -0.5 = 90 deg left, +-1 = directly behind)
#    7  LiDAR per-sector closeness (lidar.py)           -- real metric ranging
#    2  heading, as [sin(yaw), cos(yaw)]                -- the policy
#       previously had no direct way to perceive its own orientation
#    3  measured body-frame velocity [vf, vl, vz], normalized by V_MAX      -- or
#       knowledge of its own current motion, which made precise
#       braking/final-approach and body-relative steering harder to learn
#    3  current smoothed command [v_forward,v_lateral,yaw_rate], in [-1,1] --
#       lets the policy "remember" what it's already commanding without
#       relying purely on the LSTM's hidden state for that
STATE_DIM = OCC_GRID * OCC_GRID + 3 + 1 + LIDAR_NUM_SECTORS + 2 + 3 + ACTION_DIM

# --------------------------------------------------------------------------
# Physical UAV motion constraints (Eq. 10-13)
# --------------------------------------------------------------------------
V_MAX = 5.0            # m/s, forward/lateral velocity scale
OMEGA_MAX = 30.0        # deg/s, yaw rate scale (5 deg/s = 1.5 deg/step, too slow to turn toward the goal)
H_MAX = 10.0            # m, cruise altitude (AirSim NED: z = -H_MAX)

# --------------------------------------------------------------------------
# Velocity smoothing / jerk regularization
# --------------------------------------------------------------------------
ALPHA_SMOOTH = 0.1    # first-order low-pass EMA responsiveness coefficient
                        # (Table 2: alpha in {0.001, 0.005, 0.01, 0.05, 0.1};
                        # 0.005-0.01 gave the best reward / episode length trade-off)

# --------------------------------------------------------------------------
# Reward shaping constants (Section 4)
#
# Deliberately small and bounded: per-step reward stays within roughly
# [-3, +1.2] and a whole episode return within roughly [-60, +250], so the
# critic's value targets stay well-scaled. (The previous multi-band version
# let returns span -78,000..+5,000, which blew the value loss up into the
# millions and let max_grad_norm clip the actor's updates to nothing.)
# --------------------------------------------------------------------------
R_TIME = -0.01                # per-step decay, encourages reaching the goal quickly
R_SUCCESS = 100.0
R_COLLISION = -60.0         # must outweigh progress banked en route, or a late crash still nets positive
R_TIMEOUT = -60.0           # same as a collision: stalling short of the goal must not beat trying (progress banked en route is ~+100)
SUCCESS_RADIUS = 2.0          # meters
MAX_EPISODE_STEPS = 600        # shorter episodes -> more terminal-reward samples per training budget
YAW_ALIGN_COEFF = 0.1         # r_yaw = -coeff * |heading error| / 180, in [-coeff, 0]; a pure cost, so just
                              # staying alive facing the goal earns nothing
JERK_COEFF = 0.15             # r_jerk = -coeff * ||a_t - a_{t-1}||: penalizes change between consecutive raw actions

# Single smooth LiDAR barrier on the nearest frontal range (meters):
#   b(r)  = max(0, 1 - r / LIDAR_SAFE_DIST)
#   r_obs = -LIDAR_PENALTY_COEFF * max(b(front_range), LIDAR_SIDE_WEIGHT * b(any_sector_range))   in [-coeff, 0]
# Replaces the stacked camera/LiDAR danger bands, which dropped the reward
# off a cliff (-10/-25/-60 and -10/-30/-80 per step) instead of giving the
# policy a gradient to follow away from obstacles.
LIDAR_SAFE_DIST = 6.0
LIDAR_PENALTY_COEFF = 2.0
LIDAR_SIDE_WEIGHT = 0.5      # side/rear sectors count half as much as the frontal sectors

# --------------------------------------------------------------------------
# Observation scaling / altitude hold
# --------------------------------------------------------------------------
# Relative target displacement [dx, dy, dz] is divided by this (meters) and
# clipped to [-1, 1] so it sits on the same scale as every other state
# feature (the raw values reach ~165 m, drowning out the [0,1] camera/LiDAR
# features). Must be >= the largest goal distance used.
GOAL_OBS_SCALE = 200.0

# Eq. 13 altitude hold: vz = clip(-(z + H_MAX), -VZ_MAX, VZ_MAX), applied
# directly (no EMA) so altitude corrections are immediate.
VZ_MAX = 2.0
VZ_ALPHA = 0.5   # light EMA on the vertical velocity command (1.0 = no filtering) to avoid bobbing

# --------------------------------------------------------------------------
# Perception-driven safety shield (airsim_env.py only -- does NOT touch the
# reward function). When enabled it overrides the executed forward/lateral
# command using the U-Net/LiDAR danger signal, which is NOT what the policy
# sampled -- PPO then credits the policy's raw action with the shielded
# outcome (action substitution, an on-policy violation). Keep it OFF while
# training; turn it on only for deployment/evaluation if you want a
# hardware-safety wrapper (e.g. test.py experiments).
# --------------------------------------------------------------------------
ENABLE_SAFETY_SHIELD = False
AVOID_DANGER_THRESHOLD = 0.15   # eye-level occupancy (0-1) above which the shield starts engaging
AVOID_DANGER_HARD = 0.55        # above this, forward velocity is cut almost to zero regardless of the curve below
AVOID_STEER_GAIN = 1.5          # added lateral-velocity correction (toward free space) at full danger severity
AVOID_BRAKE_GAIN = 0.9          # fraction of forward velocity cut at full danger severity
AVOID_STEER_SIGN = 1.0          # flip to -1.0 if the drone is observed steering toward obstacles instead of away
AVOID_EMA_ALPHA = 0.35          # smooths the per-frame danger/steer signal before it drives the shield, so
                                # single-frame U-Net noise doesn't jerk the lateral/forward output step to step

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
ACTION_DURATION = 0.3

# Real-time stepping (default, SYNC_STEPPING = False): the simulator keeps
# running while Python reads the camera/U-Net/LiDAR and runs the policy. Each
# velocity command is sent with COMMAND_HOLD_DURATION (>> a step), without
# waiting for it, so it never expires: the drone keeps flying on its last
# command until the next one replaces it (no coast/brake between steps).
# ACTION_DURATION is then the *minimum* wall-clock time between sensor reads
# (the loop sleeps if it finishes early); real steps can be longer when
# processing is slow, and the reward's progress term uses the measured
# elapsed time per step. Processing latency is therefore part of training,
# as it will be on real hardware.
#
# SYNC_STEPPING = True instead pauses the sim between steps and advances it by
# exactly ACTION_DURATION of simulated time (simContinueForTime), for
# deterministic, latency-free stepping (only for comparison/debugging).
SYNC_STEPPING = False

# Must equal "ClockSpeed" in AirSim's settings.json (restart AirSim after
# changing it). All durations in this file (ACTION_DURATION, the reward's dt,
# COMMAND_HOLD_DURATION) are SIMULATED seconds; real-time stepping converts
# between wall-clock and sim time with this factor. Raise gradually (2, 3, ...)
# and stop at the first erratic flight/physics; the gain is limited by how fast
# the camera + U-Net + LiDAR processing runs, since that is wall-clock time.
CLOCK_SPEED = 2.0
COMMAND_HOLD_DURATION = 5.0

# --------------------------------------------------------------------------
# Target / episode sampling (AirSimNH suburban block, NED frame, meters)
# --------------------------------------------------------------------------
START_POSITION = (0.0, 0.0, -H_MAX)
GOAL_SAMPLING_RADIUS_MIN = 40.0
GOAL_SAMPLING_RADIUS_MAX = 110.0

# Fixed deployment goal, e.g. (25.0, -40.0, -15.0) in NED meters (z negative = up).
# train_ppo.py ignores this by default and samples random goals (pass --fixed_goal to use it);
# set it (or pass --goal_x/--goal_y/--goal_z / reset(options={"goal": (x,y,z)}))
# to fly to one specific destination.
# FIXED_GOAL = [80, -97.22, -0.15] # AIRSIMNH
FIXED_GOAL = [-32.57, -164.56, -0.26] # IIITS

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TOTAL_TIMESTEPS = 1_000_000
