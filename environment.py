"""
Self-Driving Car Environment for SB3 PPO Training

Uses PyBullet for physics simulation. The car drives around a
procedurally generated racetrack and learns to stay on the road.
"""

import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Any
import yaml
import importlib.util


def _import_from_path(module_name, file_path):
    """Load a python module from a file path (handles spaces in filenames)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_track_module = _import_from_path("track", os.path.join(_THIS_DIR, "setup", "track (3).py"))
_controls_module = _import_from_path("controls", os.path.join(_THIS_DIR, "setup", "controls (1).py"))
Track = _track_module.Track
TankDriveController = _controls_module.TankDriveController


class SelfDrivingCarEnv(gym.Env):
    """
    Gymnasium environment for a self-driving car on a racetrack.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config_path: str = "setup/track_config (1).yaml",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # PyBullet setup
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        gravity = self.config['physics'].get('gravity', 0.0)
        time_step = self.config['physics'].get('time_step', 1/30)
        p.setGravity(0, 0, gravity, physicsClientId=self.physics_client)
        p.setTimeStep(time_step, physicsClientId=self.physics_client)

        self.max_linear_velocity = self.config['physics']['car']['max_linear_velocity']
        self.max_angular_velocity = self.config['physics']['car']['max_angular_velocity']

        # Track
        self.track = Track(config_path)
        self.track.spawn_in_pybullet(self.physics_client)
        self.track_center_radius = (self.track.inner_radius + self.track.outer_radius) / 2

        if render_mode == "human":
            cam_dist = self.track.outer_radius * 2.5
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=0, cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0], physicsClientId=self.physics_client
            )

        # Car
        car_urdf_path = os.path.join(_THIS_DIR, "setup", "car (1).urdf")
        if not os.path.exists(car_urdf_path):
            raise FileNotFoundError(f"Car URDF not found: {car_urdf_path}")

        # Compute spawn position at midpoint of first track segment
        inner_pt = self.track.inner_points[0]
        outer_pt = self.track.outer_points[0]
        spawn_x = (inner_pt[0] + outer_pt[0]) / 2.0
        spawn_y = (inner_pt[1] + outer_pt[1]) / 2.0
        spawn_z = self.config['spawn']['position'][2]

        inner_next = self.track.inner_points[1]
        outer_next = self.track.outer_points[1]
        next_x = (inner_next[0] + outer_next[0]) / 2.0
        next_y = (inner_next[1] + outer_next[1]) / 2.0
        heading = np.arctan2(next_y - spawn_y, next_x - spawn_x)
        spawn_orn = p.getQuaternionFromEuler([0, 0, heading])

        self.car_id = p.loadURDF(
            car_urdf_path, basePosition=[spawn_x, spawn_y, spawn_z],
            baseOrientation=spawn_orn, physicsClientId=self.physics_client
        )
        self.spawn_pos = [spawn_x, spawn_y, spawn_z]
        self.spawn_orn = spawn_orn

        self.controller = TankDriveController(config_path, self.car_id, self.physics_client)

        # Action space: [throttle, steering] both in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation space: [car_x, car_y, heading_cos, heading_sin,
        #                     forward_vel, angular_vel, dist_inner, dist_outer,
        #                     dist_center, progress_angle]
        obs_low = np.full(10, -5.0, dtype=np.float32)
        obs_high = np.full(10, 5.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Episode tracking
        self.last_position = np.array([spawn_x, spawn_y])
        self.episode_reward = 0.0

    def _get_car_heading(self, car_orn):
        """Get yaw angle from quaternion."""
        return p.getEulerFromQuaternion(car_orn)[2]

    def _get_observation(self):
        """Get position-based observation vector (10-dim)."""
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, car_ang_vel = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)
        heading = self._get_car_heading(car_orn)

        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        dist_inner = dist_from_center - self.track.inner_radius
        dist_outer = self.track.outer_radius - dist_from_center
        progress = np.arctan2(car_pos[1], car_pos[0])

        forward_vel = np.sqrt(car_vel[0]**2 + car_vel[1]**2)

        obs = np.array([
            car_pos[0], car_pos[1],
            np.cos(heading), np.sin(heading),
            forward_vel / self.max_linear_velocity,
            car_ang_vel[2] / self.max_angular_velocity,
            dist_inner, dist_outer,
            dist_from_center - self.track_center_radius,
            progress,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_reward(self, action):
        """Reward: survival + velocity - off track penalty."""
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        on_track = self.track.inner_radius < dist_from_center < self.track.outer_radius

        if not on_track:
            return -50.0

        reward = 0.5  # survival bonus
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        reward += speed * 1.0  # velocity reward

        return reward

    def _is_terminated(self):
        """End episode if car is off track or flipped."""
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        dist = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        if dist < self.track.inner_radius or dist > self.track.outer_radius:
            return True
        roll, pitch, _ = p.getEulerFromQuaternion(car_orn)
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False

    def _is_truncated(self):
        return self.current_step >= self.max_episode_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        p.resetBasePositionAndOrientation(
            self.car_id, self.spawn_pos, self.spawn_orn,
            physicsClientId=self.physics_client
        )
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.physics_client)

        self.last_position = np.array(self.spawn_pos[:2])
        self.episode_reward = 0.0

        obs = self._get_observation()
        info = {"spawn_position": self.spawn_pos}
        return obs, info

    def step(self, action):
        self.current_step += 1

        forward_input = float(np.clip(action[0], -1.0, 1.0))
        turn_input = float(np.clip(action[1], -1.0, 1.0))

        linear_vel = forward_input * self.max_linear_velocity
        angular_vel = turn_input * self.max_angular_velocity

        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_forward = rot[:, 0]

        p.resetBaseVelocity(
            self.car_id,
            linearVelocity=[linear_vel * car_forward[0], linear_vel * car_forward[1], 0.0],
            angularVelocity=[0.0, 0.0, angular_vel],
            physicsClientId=self.physics_client
        )

        fixed_z = self.config['spawn']['position'][2]
        if abs(car_pos[2] - fixed_z) > 0.001:
            p.resetBasePositionAndOrientation(
                self.car_id, [car_pos[0], car_pos[1], fixed_z], car_orn,
                physicsClientId=self.physics_client
            )

        p.stepSimulation(physicsClientId=self.physics_client)

        obs = self._get_observation()
        reward = self._compute_reward(action)
        self.episode_reward += reward
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        new_pos, _ = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "car_position": new_pos,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            cam_dist = self.track.outer_radius * 2.5
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=0, cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0], physicsClientId=self.physics_client
            )
            return None
        elif self.render_mode == "rgb_array":
            cam_dist = self.track.outer_radius * 2.5
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0], distance=cam_dist,
                yaw=0, pitch=-89, roll=0, upAxisIndex=2
            )
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
            _, _, rgb, _, _ = p.getCameraImage(
                640, 480, viewMatrix=view, projectionMatrix=proj,
                physicsClientId=self.physics_client
            )
            return np.array(rgb)[:, :, :3]
        return None

    def close(self):
        if hasattr(self, 'physics_client'):
            p.disconnect(physicsClientId=self.physics_client)
