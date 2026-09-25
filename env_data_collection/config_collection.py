"""
Configuration for autonomous RGB/segmentation-mask data collection used to
build the U-Net training set (data/images, data/masks).

Edit the values in this file to match your institute AirSim environment
before running calibrate_mask_color.py and collect_data.py.
"""

import os

# --------------------------------------------------------------------------
# AirSim connection (kept independent from the top-level config.py so this
# folder can be copied/run standalone)
# --------------------------------------------------------------------------
AIRSIM_IP = "127.0.0.1"
VEHICLE_NAME = ""
CAMERA_NAME = "0"          # front FPV camera id, same as config.CAMERA_NAME
IMG_SIZE = 256              # only used for the zero-image fallback shape

# --------------------------------------------------------------------------
# Output layout - matches what train_unet.py expects one level up:
#   data/images/frame_XXXXXX.png
#   data/masks/frame_XXXXXX.png
# --------------------------------------------------------------------------
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
MASKS_DIR = os.path.join(OUTPUT_ROOT, "masks")

# Segmentation-color calibration file written by calibrate_mask_color.py and
# read by collect_data.py / mask_utils.py
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacle_color.json")

# --------------------------------------------------------------------------
# Obstacle definition (Step 1). Edit these regexes to match the actual
# object names you gave things in Unreal's World Outliner. Anything that
# does NOT match one of these patterns keeps segmentation ID 0 (free space).
# --------------------------------------------------------------------------
OBSTACLE_ID = 1
OBSTACLE_NAME_PATTERNS = [
    r"Shape_Cube.*",
    r"SM_Spotlight.*",
    r"Shape_Cylinder.*",
    r"Shape_Plane.*",
    r"Shape_Tube.*",
    r"BlockingVolume.*",
    r"SM_SuperTree.*",
    r"SM_Mobile_Trees.*",
    r"SM_Palm_01_05_G.*",
    # r"Person.*",
    # r"Fence.*",
    # r"Sign.*",
]

# --------------------------------------------------------------------------
# Flight coverage area (NED meters, relative to the PlayerStart / origin).
# Set this to bound the full campus/institute extent you want covered.
# --------------------------------------------------------------------------
X_MIN, X_MAX = -120.0, 120.0
Y_MIN, Y_MAX = -120.0, 120.0

# Lawnmower (boustrophedon) sweep spacing in meters between adjacent lines.
# Smaller = denser coverage but more waypoints/time.
GRID_SPACING = 12.0

# Altitude layers to fly the sweep at (NED z is negative-up, so these are
# entered as positive "height above start" and negated internally).
ALTITUDES = [6.0, 15.0, 30.0]

# Extra random jitter (meters) added to each waypoint's x/y so repeated
# altitude passes don't retrace pixel-identical tracks.
WAYPOINT_JITTER = 3.0

# Headings (degrees, 0 = facing +X / forward) sampled at every waypoint so
# both near/far and side/front obstacles are captured from the same spot.
YAW_SAMPLES_DEG = [0, 60, 120, 180, 240, 300]

# --------------------------------------------------------------------------
# Frame de-duplication - only save a new pair once the drone has moved this
# far (meters) or turned this much (degrees) since the last saved frame.
# --------------------------------------------------------------------------
MIN_SAVE_DIST_M = 3.0
MIN_SAVE_YAW_DEG = 20.0

# --------------------------------------------------------------------------
# Collection targets / safety
# --------------------------------------------------------------------------
TARGET_PAIRS = 8000
MOVE_VELOCITY = 6.0             # m/s used for moveToPositionAsync between waypoints
DESCENT_VELOCITY = 3.0          # m/s used for the final vertical descent leg (slower = less likely to punch through a tree canopy before the collision watchdog reacts)
MOVE_TIMEOUT_S = 25.0
YAW_SETTLE_S = 0.6              # pause after rotating so the render settles before capture
COLLISION_RECOVERY_ALT_M = 40.0  # climb to this height (positive, meters) after a collision

# Horizontal transit between waypoints happens at this altitude (positive
# meters), which must clear the tallest obstacle (tree/building) in the
# scene. The drone only descends to a layer's real altitude once it is
# already above the target x/y, so it never has to punch sideways through
# a tree to get from one waypoint to the next -- that "diving through the
# canopy" straight-line behaviour is what previously got it stuck.
CRUISE_ALT_M = max(ALTITUDES) + 15.0

# Collision handling during the move-to-waypoint / descend legs.
COLLISION_POLL_INTERVAL_S = 0.15   # how often to check simGetCollisionInfo while a move is in flight
COLLISION_BACKOFF_M = 6.0          # distance to retreat along the collision surface normal
MAX_COLLISIONS_PER_CELL = 2        # after this many collisions near the same grid cell, stop retrying it for the rest of the run

# If True, also fires a bonus randomized top-up pass (jittered positions
# drawn from the whole bounding box + altitude list) after the main
# lawnmower sweep finishes, until TARGET_PAIRS is reached.
RANDOM_TOPUP = True

# --------------------------------------------------------------------------
# Train/validation split by route block (Step 3). The lawnmower sweep is
# divided into columns; columns whose index falls in VAL_COLUMN_INDICES are
# written to data_val/ instead of data/ so validation frames come from a
# spatially distinct area rather than randomly-split adjacent frames.
# --------------------------------------------------------------------------
VAL_SPLIT_ENABLED = True
VAL_COLUMN_MODULO = 6   # every Nth sweep column is held out
VAL_COLUMN_REMAINDER = 5
VAL_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_val")
VAL_IMAGES_DIR = os.path.join(VAL_OUTPUT_ROOT, "images")
VAL_MASKS_DIR = os.path.join(VAL_OUTPUT_ROOT, "masks")
