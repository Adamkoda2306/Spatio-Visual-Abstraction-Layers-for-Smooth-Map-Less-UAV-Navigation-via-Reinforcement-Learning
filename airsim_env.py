"""
Gymnasium environment wrapping AirSim (AirSimNH, Multirotor mode) that implements
the spatio-visual navigation pipeline from Section 3 of the paper:

  camera -> U-Net -> 5x5 occupancy compression -> 28-dim state
  -> continuous velocity action -> exponential command smoothing -> AirSim execution
  -> multi-modal reward (Section 4)

Control: the action is [v_forward, v_lateral, yaw_rate] in [-1, 1] (Eq. 9),
executed through AirSim's `moveByVelocityBodyFrameAsync` (Section 3.4).
Altitude is held automatically by the Eq. 13 cruise-altitude controller
(vz = -(z + H_MAX)), so the policy only has to learn the 3 navigation axes
the paper actually optimizes over.

Goal: pass an explicit (x, y, z) NED destination via the constructor, via
`reset(options={"goal": (x, y, z)})`, or via config.FIXED_GOAL. If none is
given, a goal is sampled randomly (useful for training generalizable policies).
"""

import math

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

import airsim

import config
from unet import ObstaclePerceptionModule
from occupancy import pool_to_grid, flatten_grid, avoidance_steer
from reward import compute_reward


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
        self._last_pos = np.array(config.START_POSITION, dtype=np.float32)
        self._prev_dist = None
        self._step_count = 0
        self._last_grid = np.zeros((config.OCC_GRID, config.OCC_GRID), dtype=np.float32)
        self._danger_ema = 0.0
        self._steer_ema = 0.0
        self._collision_baseline_ts = 0
        self.trajectory = []  # populated during episodes for plotting/analysis

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

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
        self._step_count = 0
        self._danger_ema = 0.0
        self._steer_ema = 0.0
        self.trajectory = []

        pos, yaw_deg = self._get_pose()
        self._last_pos = pos.copy()
        dx, dy, dz = self._relative_target(pos)
        self._prev_dist = math.sqrt(dx ** 2 + dy ** 2)

        grid = self._get_occupancy_grid()
        self._last_grid = grid
        obs = self._build_state(grid, dx, dy, dz)
        self.trajectory.append(pos.copy())

        return obs.astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a_forward, a_lateral, a_yaw_rate = action

        # Eq. 23: first-order exponential smoothing of the raw velocity
        # command, so stick inputs don't change abruptly.
        cmd_target = action
        self._cmd_smooth = (1 - config.ALPHA_SMOOTH) * self._cmd_smooth + config.ALPHA_SMOOTH * cmd_target

        # Perception-driven safety shield: softens the *executed*
        # forward/lateral velocity using the U-Net occupancy grid from what
        # the drone currently sees (self._last_grid, set at the end of the
        # previous step/reset), so an action that would fly straight into an
        # obstacle gets steered toward the free space in the probability map
        # instead of relying purely on the reward gradient to learn that
        # over millions of steps. The reward below still scores the agent's
        # own raw action (a_yaw_rate, a_lateral) exactly as before -- this
        # only changes what actually gets flown.
        #
        # danger/steer are EMA-smoothed across steps before being used: a raw
        # per-frame U-Net reading is noisy (lighting flicker, a leaf moving,
        # slight pose jitter), and reacting to every single-frame blip is
        # exactly what made the drone visibly shake in 3rd-person view. The
        # EMA alpha is deliberately much faster than ALPHA_SMOOTH (which
        # smooths the *raw action*, not this safety signal), so real,
        # persistent obstacles are still reacted to within a few steps.
        raw_steer, raw_danger = avoidance_steer(self._last_grid)
        self._danger_ema += config.AVOID_EMA_ALPHA * (raw_danger - self._danger_ema)
        self._steer_ema += config.AVOID_EMA_ALPHA * (raw_steer - self._steer_ema)
        danger, steer_lr = self._danger_ema, self._steer_ema

        exec_forward, exec_lateral = float(self._cmd_smooth[0]), float(self._cmd_smooth[1])
        if danger > config.AVOID_DANGER_THRESHOLD:
            severity = min(1.0, (danger - config.AVOID_DANGER_THRESHOLD) / (1.0 - config.AVOID_DANGER_THRESHOLD))
            severity = severity ** 0.5  # react hard early rather than waiting for danger -> 1.0
            exec_lateral = float(np.clip(
                exec_lateral + config.AVOID_STEER_SIGN * config.AVOID_STEER_GAIN * severity * steer_lr,
                -1.0, 1.0,
            ))
            exec_forward = float(exec_forward * (1.0 - config.AVOID_BRAKE_GAIN * severity))

        if danger > config.AVOID_DANGER_HARD:
            # something fills the center of the frame right now -- stop
            # closing distance on it almost entirely regardless of the
            # smooth curve above, and let the lateral correction do the dodging.
            exec_forward = float(exec_forward * 0.05)

        v_forward = exec_forward * config.V_MAX
        v_lateral = exec_lateral * config.V_MAX
        yaw_rate_deg = float(self._cmd_smooth[2]) * config.OMEGA_MAX

        # Eq. 13: hold the cruise altitude H_MAX using the z recorded at the
        # end of the previous step (avoids an extra AirSim round-trip here).
        vz = float(np.clip(-(self._last_pos[2] + config.H_MAX), -config.V_MAX, config.V_MAX))

        self.client.moveByVelocityBodyFrameAsync(
            v_forward,
            v_lateral,
            vz,
            config.ACTION_DURATION,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
            vehicle_name=self.vehicle_name,
        ).join()

        self._step_count += 1

        pos, yaw_deg = self._get_pose()
        self._last_pos = pos.copy()
        dx, dy, dz = self._relative_target(pos)
        d_curr = math.sqrt(dx ** 2 + dy ** 2)

        grid = self._get_occupancy_grid()
        self._last_grid = grid
        obs = self._build_state(grid, dx, dy, dz)

        body_velocity = self._get_body_velocity(yaw_deg)

        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        collided = bool(collision_info.has_collided and collision_info.time_stamp != self._collision_baseline_ts)
        timed_out = self._step_count >= config.MAX_EPISODE_STEPS

        reward, info = compute_reward(
            d_prev=self._prev_dist,
            d_curr=d_curr,
            current_yaw_deg=yaw_deg,
            dx=dx,
            dy=dy,
            yaw_rate_norm=a_yaw_rate,
            lateral_norm=a_lateral,
            body_velocity=body_velocity,
            cmd_smooth=self._cmd_smooth,
            cmd_target=cmd_target,
            occupancy_grid=grid,
            collided=collided,
            timed_out=timed_out,
        )

        self._prev_dist = d_curr
        self.trajectory.append(pos.copy())

        success = d_curr < config.SUCCESS_RADIUS
        terminated = bool(success or collided)
        truncated = bool(timed_out and not terminated)

        info.update(dict(success=success, collided=collided, timed_out=timed_out, distance=d_curr))

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
        except Exception:
            pass

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

    def _build_state(self, grid, dx, dy, dz):
        v_visual = flatten_grid(grid)  # Eq. 5, 25-dim
        return np.concatenate([v_visual, [dx, dy, dz]])  # Eq. 7, 28-dim
