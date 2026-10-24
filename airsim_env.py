"""
Gymnasium environment wrapping AirSim (AirSimNH, Multirotor mode) that implements
the spatio-visual navigation pipeline from Section 3 of the paper:

  camera -> U-Net -> 5x5 occupancy compression -> 28-dim state
  -> continuous velocity action -> exponential command smoothing -> AirSim execution
  -> bounded reward (reward.py)

Control: the action is [v_forward, v_lateral, yaw_rate] in [-1, 1] (Eq. 9),
executed through AirSim's `moveByVelocityBodyFrameAsync` (Section 3.4).
Altitude is held automatically by the Eq. 13 cruise-altitude controller
(vz = -(z + H_MAX)), so the policy only has to learn the 3 navigation axes
the paper actually optimizes over.

Perception is dual-sensor: the camera/U-Net occupancy grid (Eq. 3-5) gives a
wide-FOV, semantic read of where free space and obstacles roughly are; the
LiDAR sensor (lidar.py) gives real metric range in meters for how close the
nearest obstacle actually is. Both feed the state; LiDAR range also feeds
the reward (reward.py). The optional safety shield (config.ENABLE_SAFETY_SHIELD,
OFF by default) is the only thing that can alter the policy's executed action.

Goal: pass an explicit (x, y, z) NED destination via the constructor, via
`reset(options={"goal": (x, y, z)})`, or via config.FIXED_GOAL. If none is
given, a goal is sampled randomly (useful for training generalizable policies).
"""

import math
import time

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

import airsim

import config
from unet import ObstaclePerceptionModule
from occupancy import pool_to_grid, flatten_grid, avoidance_steer
from reward import compute_reward
import live_view
import lidar as lidar_mod


class AirSimUAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ip=config.AIRSIM_IP, vehicle_name=config.VEHICLE_NAME,
                 perception=None, goal=None):
        super().__init__()

        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        self.perception = perception or ObstaclePerceptionModule()

        # explicit goal takes precedence over config.FIXED_GOAL, both take
        # precedence over the per-episode random sampling done in reset()
        self._fixed_goal = np.array(goal, dtype=np.float32) if goal is not None else (
            np.array(config.FIXED_GOAL, dtype=np.float32) if config.FIXED_GOAL is not None else None
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(config.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.STATE_DIM,), dtype=np.float32
        )

        self._goal = np.zeros(3, dtype=np.float32)
        self._cmd_smooth = np.zeros(config.ACTION_DIM, dtype=np.float32)  # [v_forward,v_lateral,yaw_rate]
        self._prev_action = np.zeros(config.ACTION_DIM, dtype=np.float32)
        self._vz_smooth = 0.0
        self._last_sense_time = time.perf_counter()  # wall-clock time of the previous sensor read
        self._last_pos = np.array(config.START_POSITION, dtype=np.float32)
        self._prev_dist = None
        self._step_count = 0
        self._last_grid = np.zeros((config.OCC_GRID, config.OCC_GRID), dtype=np.float32)
        self._danger_ema = 0.0
        self._steer_ema = 0.0
        self._last_lidar_closeness = np.zeros(config.LIDAR_NUM_SECTORS, dtype=np.float32)
        self._last_lidar_ranges = np.full(config.LIDAR_NUM_SECTORS, config.LIDAR_MAX_RANGE, dtype=np.float32)
        self._danger_ema_l = 0.0
        self._steer_ema_l = 0.0
        self._collision_baseline_ts = 0
        self.trajectory = []  # populated during episodes for plotting/analysis

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # takeoff/climb below need simulated time to run, so un-pause for the reset
        # sequence and re-pause at its end (see config.SYNC_STEPPING).
        if config.SYNC_STEPPING:
            self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        goal_override = (options or {}).get("goal") if options else None
        if goal_override is not None:
            self._goal = np.array(goal_override, dtype=np.float32)
        elif self._fixed_goal is not None:
            self._goal = self._fixed_goal.copy()
        else:
            self._goal = self._sample_goal()

        # Climb to the cruise altitude H_MAX: per Eq. 13, altitude is held
        # automatically at -H_MAX for the whole episode (vz = -(z + H_MAX)
        # is recomputed every step in step()), not commanded by the policy.
        # Success/distance (Eq. 17-18) only depend on horizontal [dx, dy],
        # so the agent only has to learn the 3 navigation axes the paper
        # actually optimizes over -- it never has to fly near ground-level
        # clutter (benches, cars, curbs) to "reach" a low-altitude goal.
        self.client.moveToZAsync(-config.H_MAX, 2.0, vehicle_name=self.vehicle_name).join()

        # AirSim's collision flag can still be set from whatever happened
        # right up to this reset (or from the climb above, if it clips
        # something); record its timestamp so step() only reacts to a
        # genuinely NEW collision recorded during this episode, not stale
        # leftover state from before the episode actually started.
        self._collision_baseline_ts = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp

        self._cmd_smooth = np.zeros(config.ACTION_DIM, dtype=np.float32)
        self._prev_action = np.zeros(config.ACTION_DIM, dtype=np.float32)
        self._vz_smooth = 0.0
        self._step_count = 0
        self._danger_ema = 0.0
        self._steer_ema = 0.0
        self._danger_ema_l = 0.0
        self._steer_ema_l = 0.0
        self.trajectory = []

        if config.SYNC_STEPPING:
            self.client.simPause(True)

        pos, yaw_deg = self._get_pose()
        self._last_sense_time = time.perf_counter()
        self._last_pos = pos.copy()
        dx, dy, dz = self._relative_target(pos)
        self._prev_dist = math.sqrt(dx ** 2 + dy ** 2)

        grid = self._get_occupancy_grid()
        self._last_grid = grid
        lidar_closeness, lidar_ranges = self._get_lidar_data()
        self._last_lidar_closeness = lidar_closeness
        self._last_lidar_ranges = lidar_ranges
        body_velocity = self._get_body_velocity(yaw_deg)
        obs = self._build_state(grid, dx, dy, dz, lidar_closeness, yaw_deg, body_velocity, self._cmd_smooth)
        self.trajectory.append(pos.copy())

        self._show_live_view(grid, yaw_deg, dx, dy, 0.0, self._prev_dist)

        return obs.astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Eq. 23: first-order exponential smoothing of the raw velocity
        # command, so stick inputs don't change abruptly.
        cmd_target = action
        self._cmd_smooth = (1 - config.ALPHA_SMOOTH) * self._cmd_smooth + config.ALPHA_SMOOTH * cmd_target

        exec_forward, exec_lateral = float(self._cmd_smooth[0]), float(self._cmd_smooth[1])
        if config.ENABLE_SAFETY_SHIELD:
            exec_forward, exec_lateral = self._apply_safety_shield(exec_forward, exec_lateral)

        v_forward = exec_forward * config.V_MAX
        v_lateral = exec_lateral * config.V_MAX
        yaw_rate_deg = float(self._cmd_smooth[2]) * config.OMEGA_MAX

        # Eq. 13: hold the cruise altitude H_MAX using the z recorded at the
        # end of the previous step (avoids an extra AirSim round-trip here).
        # Proportional hold with only a light EMA (VZ_ALPHA): altitude is not the
        # policy's job, so it must respond quickly, but unfiltered it bobbed.
        vz_target = float(np.clip(-(self._last_pos[2] + config.H_MAX), -config.VZ_MAX, config.VZ_MAX))
        self._vz_smooth = (1 - config.VZ_ALPHA) * self._vz_smooth + config.VZ_ALPHA * vz_target
        vz = float(self._vz_smooth)

        # Long-lived command, not waited on: the drone keeps flying on it until
        # the next step's command replaces it, so it never coasts/brakes while
        # Python processes sensors.
        self.client.moveByVelocityBodyFrameAsync(
            v_forward,
            v_lateral,
            vz,
            config.COMMAND_HOLD_DURATION,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
            vehicle_name=self.vehicle_name,
        )
        if config.SYNC_STEPPING:
            # advance exactly ACTION_DURATION of sim time, then freeze again
            self.client.simContinueForTime(config.ACTION_DURATION)
            dt = config.ACTION_DURATION
        else:
            # let the command act for at least ACTION_DURATION since the last
            # sensor read; if policy + processing already took longer, don't wait
            remaining = config.ACTION_DURATION - (time.perf_counter() - self._last_sense_time)
            if remaining > 0:
                time.sleep(remaining)

        self._step_count += 1

        pos, yaw_deg = self._get_pose()
        t_pose = time.perf_counter()  # the position (hence d_curr) is as of this moment
        self._last_pos = pos.copy()
        dx, dy, dz = self._relative_target(pos)
        d_curr = math.sqrt(dx ** 2 + dy ** 2)

        grid = self._get_occupancy_grid()
        self._last_grid = grid

        # Fresh LiDAR reading for this step's state/reward.
        lidar_closeness, lidar_ranges = self._get_lidar_data()
        self._last_lidar_closeness = lidar_closeness
        self._last_lidar_ranges = lidar_ranges
        lidar_front_m, lidar_global_m = lidar_mod.frontal_and_global_range(lidar_ranges)

        body_velocity = self._get_body_velocity(yaw_deg)

        obs = self._build_state(grid, dx, dy, dz, lidar_closeness, yaw_deg, body_velocity, self._cmd_smooth)

        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        collided = bool(collision_info.has_collided and collision_info.time_stamp != self._collision_baseline_ts)
        timed_out = self._step_count >= config.MAX_EPISODE_STEPS

        if not config.SYNC_STEPPING:
            dt = t_pose - self._last_sense_time
            self._last_sense_time = t_pose

        reward, info = compute_reward(
            d_prev=self._prev_dist,
            d_curr=d_curr,
            current_yaw_deg=yaw_deg,
            dx=dx,
            dy=dy,
            action=action,
            prev_action=self._prev_action,
            occupancy_grid=grid,
            collided=collided,
            timed_out=timed_out,
            dt=dt,
            lidar_front_m=lidar_front_m,
            lidar_global_m=lidar_global_m,
        )

        self._prev_dist = d_curr
        self._prev_action = action.copy()
        self.trajectory.append(pos.copy())

        success = d_curr < config.SUCCESS_RADIUS
        terminated = bool(success or collided)
        truncated = bool(timed_out and not terminated)

        # "TimeLimit.truncated" makes SB3 bootstrap the value at a timeout
        # instead of treating it as a true terminal state.
        info.update(dict(success=success, collided=collided, timed_out=timed_out, distance=d_curr,
                         **{"TimeLimit.truncated": truncated}))
        if collided:
            info["collision_object"] = collision_info.object_name

        self._show_live_view(grid, yaw_deg, dx, dy, exec_lateral, d_curr)

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _apply_safety_shield(self, exec_forward, exec_lateral):
        """Optional perception-driven shield (config.ENABLE_SAFETY_SHIELD):
        brakes and steers the *executed* command toward free space using the
        previous step's U-Net grid and LiDAR closeness. Never enable during
        RL training -- the rollout buffer keeps the policy's raw action, so
        shield rescues get credited to actions that would have crashed."""
        # danger/steer are EMA-smoothed across steps: a raw per-frame U-Net
        # reading is noisy and reacting to every blip makes the drone shake.
        raw_steer, raw_danger = avoidance_steer(self._last_grid)
        self._danger_ema += config.AVOID_EMA_ALPHA * (raw_danger - self._danger_ema)
        self._steer_ema += config.AVOID_EMA_ALPHA * (raw_steer - self._steer_ema)

        raw_steer_l, raw_danger_l = lidar_mod.steer_from_closeness(self._last_lidar_closeness)
        self._danger_ema_l += config.AVOID_EMA_ALPHA * (raw_danger_l - self._danger_ema_l)
        self._steer_ema_l += config.AVOID_EMA_ALPHA * (raw_steer_l - self._steer_ema_l)

        # Worse of the two danger readings; steering favors LiDAR.
        danger = max(self._danger_ema, self._danger_ema_l)
        steer_lr = (
            config.LIDAR_STEER_WEIGHT * self._steer_ema_l
            + (1.0 - config.LIDAR_STEER_WEIGHT) * self._steer_ema
        )

        if danger > config.AVOID_DANGER_THRESHOLD:
            severity = min(1.0, (danger - config.AVOID_DANGER_THRESHOLD) / (1.0 - config.AVOID_DANGER_THRESHOLD))
            severity = severity ** 0.5  # react hard early rather than waiting for danger -> 1.0
            exec_lateral = float(np.clip(
                exec_lateral + config.AVOID_STEER_SIGN * config.AVOID_STEER_GAIN * severity * steer_lr,
                -1.0, 1.0,
            ))
            exec_forward = float(exec_forward * (1.0 - config.AVOID_BRAKE_GAIN * severity))

        if danger > config.AVOID_DANGER_HARD:
            exec_forward = float(exec_forward * 0.05)
        return exec_forward, exec_lateral

    def close(self):
        try:
            if config.SYNC_STEPPING:
                self.client.simPause(False)
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
        except Exception:
            pass
        live_view.close()

    def _show_live_view(self, grid, yaw_deg, dx, dy, exec_lateral, distance):
        """Update the live occupancy-map HUD (live_view.py): marks the grid
        column the goal bearing projects onto ("GOAL") and the column the
        currently executed command is steering toward ("GOING")."""
        bearing_deg = (math.degrees(math.atan2(dy, dx)) - yaw_deg + 180.0) % 360.0 - 180.0
        goal_col, goal_visible = live_view.goal_bearing_to_grid_col(bearing_deg)
        action_col = (exec_lateral + 1.0) / 2.0 * (config.OCC_GRID - 1)
        live_view.show(grid, goal_col, goal_visible, action_col, distance, self._step_count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_goal(self, x, y, z):
        """Set an explicit NED destination to use on the next reset()."""
        self._fixed_goal = np.array([x, y, z], dtype=np.float32)

    def _sample_goal(self):
        angle = np.random.uniform(0, 2 * math.pi)
        radius = np.random.uniform(config.GOAL_SAMPLING_RADIUS_MIN, config.GOAL_SAMPLING_RADIUS_MAX)
        gx = config.START_POSITION[0] + radius * math.cos(angle)
        gy = config.START_POSITION[1] + radius * math.sin(angle)
        gz = -config.H_MAX
        return np.array([gx, gy, gz], dtype=np.float32)

    def _get_pose(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        _, _, yaw_rad = airsim.to_eularian_angles(orientation)
        pos_arr = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        return pos_arr, math.degrees(yaw_rad)

    def _get_body_velocity(self, yaw_deg):
        """Rotate the measured NED world-frame linear velocity into the
        vehicle's body frame [v_forward, v_lateral, v_z] using current yaw."""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        v = state.kinematics_estimated.linear_velocity
        yaw = math.radians(yaw_deg)
        v_forward = v.x_val * math.cos(yaw) + v.y_val * math.sin(yaw)
        v_lateral = -v.x_val * math.sin(yaw) + v.y_val * math.cos(yaw)
        return np.array([v_forward, v_lateral, v.z_val], dtype=np.float32)

    def _relative_target(self, pos):
        drel = self._goal - pos  # Eq. 6
        return float(drel[0]), float(drel[1]), float(drel[2])

    def _get_rgb_image(self):
        raw = self.client.simGetImage(config.CAMERA_NAME, airsim.ImageType.Scene, vehicle_name=self.vehicle_name)
        if raw is None:
            return np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        png = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if png is None:
            return np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        return cv2.cvtColor(png, cv2.COLOR_BGR2RGB)

    def _get_occupancy_grid(self):
        rgb = self._get_rgb_image()
        prob_map = self.perception.predict(rgb)  # Eq. 3
        grid = pool_to_grid(prob_map)             # Eq. 4
        return grid

    def _get_lidar_data(self):
        """Reads the LiDAR sensor and returns (closeness, ranges_m):
        closeness is an (LIDAR_NUM_SECTORS,) array in [0,1] (0=clear,
        1=very close), for the state vector; ranges_m is the same data in
        real meters, for the reward/safety-shield metric-distance logic."""
        lidar_data = self.client.getLidarData(lidar_name=config.LIDAR_NAME, vehicle_name=self.vehicle_name)
        points = lidar_mod.parse_point_cloud(lidar_data.point_cloud)
        ranges_m = lidar_mod.sector_ranges(points)
        closeness = lidar_mod.closeness_from_ranges(ranges_m)
        return closeness, ranges_m

    def _build_state(self, grid, dx, dy, dz, lidar_closeness, yaw_deg, body_velocity, cmd_smooth):
        v_visual = flatten_grid(grid)  # Eq. 5, 25-dim
        yaw_rad = math.radians(yaw_deg)
        heading = np.array([math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float32)
        goal_rel = np.clip(np.array([dx, dy, dz], dtype=np.float32) / config.GOAL_OBS_SCALE, -1.0, 1.0)
        # signed heading error to the goal, wrapped to [-pi, pi) and scaled to [-1, 1)
        bearing = (math.atan2(dy, dx) - yaw_rad + math.pi) % (2 * math.pi) - math.pi
        goal_bearing = np.array([bearing / math.pi], dtype=np.float32)
        vel_norm = np.clip(np.asarray(body_velocity, dtype=np.float32) / config.V_MAX, -1.0, 1.0)
        return np.concatenate([
            v_visual,                                   # 25: camera/U-Net occupancy (Eq. 5)
            goal_rel,                                    # 3:  relative target displacement (Eq. 6), scaled to [-1,1]
            goal_bearing,                                # 1:  goal bearing relative to heading, [-1,1]
            lidar_closeness.astype(np.float32),          # 7:  LiDAR per-sector closeness
            heading,                                     # 2:  sin(yaw), cos(yaw)
            vel_norm,                                    # 3:  measured body velocity, normalized
            np.asarray(cmd_smooth, dtype=np.float32),    # 3:  current smoothed command
        ])
