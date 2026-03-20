"""
Custom Gymnasium environment for training a self-driving RC car using PPO.

The car navigates a procedurally generated racetrack in PyBullet, receiving
sensor-realistic observations (IMU + LiDAR) and learning to stay on the road
while making forward progress.
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
    """Load a Python module from an arbitrary file path (handles spaces in filenames)."""
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
    Gymnasium environment for an RL-controlled RC car on a racetrack.

    Observation space (13-dimensional):
        [0]  cos(heading)           -- IMU compass
        [1]  sin(heading)           -- IMU compass
        [2]  normalized forward vel -- wheel encoder
        [3]  normalized angular vel -- IMU gyroscope
        [4-12] 9 LiDAR distance rays spanning the front 180 degrees

    Action space (2-dimensional, continuous [-1, 1]):
        [0] throttle (forward / backward)
        [1] steering (left / right)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # LiDAR configuration
    LIDAR_ANGLES_DEG = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
    LIDAR_MAX_RANGE = 5.0
    LIDAR_NUM_RAYS = 9

    def __init__(
        self,
        config_path: str = "setup/track_config (1).yaml",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 3000,
    ):
        super().__init__()

        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # --- PyBullet physics setup ---
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        gravity = self.config["physics"].get("gravity", 0.0)
        time_step = self.config["physics"].get("time_step", 1 / 30)
        p.setGravity(0, 0, gravity, physicsClientId=self.physics_client)
        p.setTimeStep(time_step, physicsClientId=self.physics_client)

        self.max_linear_velocity = self.config["physics"]["car"]["max_linear_velocity"]
        self.max_angular_velocity = self.config["physics"]["car"]["max_angular_velocity"]

        # --- Track ---
        self.track = Track(config_path)
        self.track.spawn_in_pybullet(self.physics_client)
        self.track_center_radius = (self.track.inner_radius + self.track.outer_radius) / 2

        if render_mode == "human":
            cam_dist = self.track.outer_radius * 2.5
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=0, cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0], physicsClientId=self.physics_client,
            )

        # --- Car ---
        car_urdf_path = os.path.join(_THIS_DIR, "setup", "car (1).urdf")
        if not os.path.exists(car_urdf_path):
            raise FileNotFoundError(f"Car URDF not found: {car_urdf_path}")

        spawn_pos, spawn_orn = self._compute_spawn_pose()
        self.car_id = p.loadURDF(
            car_urdf_path, basePosition=spawn_pos, baseOrientation=spawn_orn,
            physicsClientId=self.physics_client,
        )
        self.spawn_pos = spawn_pos
        self.spawn_orn = spawn_orn

        self.controller = TankDriveController(config_path, self.car_id, self.physics_client)

        # --- Spaces ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        obs_low = np.array([-1.0, -1.0, -1.0, -1.0] + [0.0] * 9, dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0] + [1.0] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- Episode tracking ---
        self.last_position = np.array(spawn_pos[:2])
        self.last_angle = 0.0
        self.total_progress = 0.0
        self.episode_reward = 0.0

        # --- Debug visuals ---
        self._debug_items = {}
        self._last_step_reward = 0.0

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _compute_spawn_pose(self):
        """Place the car at the midpoint of the first track segment, facing forward."""
        inner_pt = self.track.inner_points[0]
        outer_pt = self.track.outer_points[0]
        spawn_x = (inner_pt[0] + outer_pt[0]) / 2.0
        spawn_y = (inner_pt[1] + outer_pt[1]) / 2.0
        spawn_z = self.config["spawn"]["position"][2]

        inner_next = self.track.inner_points[1]
        outer_next = self.track.outer_points[1]
        next_x = (inner_next[0] + outer_next[0]) / 2.0
        next_y = (inner_next[1] + outer_next[1]) / 2.0

        heading = np.arctan2(next_y - spawn_y, next_x - spawn_x)
        spawn_orn = p.getQuaternionFromEuler([0, 0, heading])
        return [spawn_x, spawn_y, spawn_z], spawn_orn

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------

    def _get_car_heading(self, car_orn):
        """Extract yaw angle (radians) from a quaternion orientation."""
        return p.getEulerFromQuaternion(car_orn)[2]

    def _cast_lidar_rays(self, car_pos_2d, heading):
        """
        Cast 9 LiDAR rays across the front 180 degrees and return
        normalized hit distances in [0, 1].
        """
        distances = np.ones(self.LIDAR_NUM_RAYS, dtype=np.float32)

        for i in range(self.LIDAR_NUM_RAYS):
            ray_angle = heading + np.radians(self.LIDAR_ANGLES_DEG[i])
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

            min_t = self.LIDAR_MAX_RANGE
            for boundary in [self.track.inner_points[:, :2], self.track.outer_points[:, :2]]:
                n = len(boundary)
                for j in range(n):
                    k = (j + 1) % n
                    t = self._ray_segment_intersection(car_pos_2d, ray_dir, boundary[j], boundary[k])
                    if t is not None and t < min_t:
                        min_t = t

            distances[i] = min(min_t / self.LIDAR_MAX_RANGE, 1.0)

        return distances

    def _ray_segment_intersection(self, ray_origin, ray_dir, seg_a, seg_b):
        """
        2-D ray-segment intersection test.

        Returns the distance along the ray to the hit point, or None if the
        ray does not intersect the segment.
        """
        seg_d = seg_b - seg_a
        cross = ray_dir[0] * seg_d[1] - ray_dir[1] * seg_d[0]

        if abs(cross) < 1e-10:
            return None

        diff = seg_a - ray_origin
        t = (diff[0] * seg_d[1] - diff[1] * seg_d[0]) / cross
        s = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / cross

        if t > 0.01 and 0 <= s <= 1:
            return t
        return None

    # ------------------------------------------------------------------
    # Track geometry helpers
    # ------------------------------------------------------------------

    def _get_progress_angle(self, car_pos):
        """Angle (radians) from the track origin to the car -- used for lap progress."""
        return np.arctan2(car_pos[1], car_pos[0])

    def _get_distances_to_track(self, car_pos):
        """
        Return (dist_to_inner_boundary, dist_to_outer_boundary, closest_segment_idx).
        """
        pos_2d = np.array([car_pos[0], car_pos[1]])

        inner_pts = self.track.inner_points[:, :2]
        inner_diffs = inner_pts - pos_2d
        inner_dists = np.sqrt(np.sum(inner_diffs ** 2, axis=1))
        closest_idx = int(np.argmin(inner_dists))
        dist_inner = inner_dists[closest_idx]

        outer_pts = self.track.outer_points[:, :2]
        outer_diffs = outer_pts - pos_2d
        outer_dists = np.sqrt(np.sum(outer_diffs ** 2, axis=1))
        dist_outer = float(np.min(outer_dists))

        return dist_inner, dist_outer, closest_idx

    def _is_on_track(self, car_pos):
        """
        Check whether the car is on the drivable surface.

        The track is an annular region. The car must be inside the larger
        boundary polygon but outside the smaller one.

        Note: Track.inner_points is the *outer* boundary and
        Track.outer_points is the *inner* boundary due to how offset normals
        are computed in the Track class.
        """
        pos_2d = np.array([car_pos[0], car_pos[1]])
        inside_outer = self._point_in_polygon(pos_2d, self.track.inner_points[:, :2])
        inside_inner = self._point_in_polygon(pos_2d, self.track.outer_points[:, :2])
        return inside_outer and not inside_inner

    def _point_in_polygon(self, point, polygon):
        """Ray-casting point-in-polygon test."""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if (yi > y) != (yj > y):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_intersect:
                    inside = not inside
            j = i
        return inside

    def _find_heading_boundary_intersection(self, car_pos_2d, heading_dir):
        """Cast a ray from the car in its heading direction and return the first boundary hit point."""
        min_t = float("inf")
        hit_point = None
        for boundary in [self.track.inner_points[:, :2], self.track.outer_points[:, :2]]:
            n = len(boundary)
            for i in range(n):
                j = (i + 1) % n
                t = self._ray_segment_intersection(car_pos_2d, heading_dir, boundary[i], boundary[j])
                if t is not None and t < min_t:
                    min_t = t
                    hit_point = car_pos_2d + heading_dir * t
        return hit_point

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self):
        """
        Build the 13-D observation vector from simulated sensors.

        IMU  (4 values): heading cos/sin, forward velocity, angular velocity
        LiDAR (9 values): normalized distances from 9 rays
        """
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, car_ang_vel = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        heading = self._get_car_heading(car_orn)

        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_forward = rot[:, 0]
        forward_vel = car_vel[0] * car_forward[0] + car_vel[1] * car_forward[1]
        angular_vel = car_ang_vel[2]

        norm_fwd = np.clip(forward_vel / self.max_linear_velocity, -1.0, 1.0)
        norm_ang = np.clip(angular_vel / self.max_angular_velocity, -1.0, 1.0)

        lidar = self._cast_lidar_rays(np.array([car_pos[0], car_pos[1]]), heading)

        observation = np.concatenate([
            np.array([np.cos(heading), np.sin(heading), norm_fwd, norm_ang], dtype=np.float32),
            lidar,
        ])
        return observation

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action):
        """
        Reward function (iteration 8 -- see log.md for full history).

        Components:
            - Angular progress around the track (+50x delta_angle) -- primary driver
            - Centering bonus (up to +0.1) -- stay away from walls
            - Off-track penalty (-10) -- episode also ends, costing future rewards
        """
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        if not self._is_on_track(car_pos):
            return -10.0

        # -- Angular progress --
        current_angle = self._get_progress_angle(car_pos)
        delta_angle = current_angle - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2.0 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2.0 * np.pi

        reward = delta_angle * 50.0

        self.last_angle = current_angle
        self.total_progress += delta_angle

        # -- Centering bonus --
        dist_inner, dist_outer, _ = self._get_distances_to_track(car_pos)
        min_edge_dist = min(dist_inner, dist_outer)
        half_width = self.track.track_width / 2.0
        centering = np.clip(min_edge_dist / half_width, 0.0, 1.0)
        reward += centering * 0.1

        return reward

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _is_terminated(self):
        """Episode ends if the car leaves the track or flips over."""
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)

        if not self._is_on_track(car_pos):
            return True

        roll, pitch, _ = p.getEulerFromQuaternion(car_orn)
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True

        return False

    def _is_truncated(self):
        """Episode ends after max_episode_steps regardless of performance."""
        return self.current_step >= self.max_episode_steps

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)
        self.current_step = 0

        p.resetBasePositionAndOrientation(
            self.car_id, self.spawn_pos, self.spawn_orn,
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.physics_client)

        self.last_position = np.array(self.spawn_pos[:2])
        self.last_angle = self._get_progress_angle(self.spawn_pos)
        self.total_progress = 0.0
        self.episode_reward = 0.0

        observation = self._get_observation()
        info = {
            "spawn_position": self.spawn_pos,
            "track_inner_radius": self.track.inner_radius,
            "track_outer_radius": self.track.outer_radius,
        }
        return observation, info

    def step(self, action):
        """
        Execute one simulation step.

        1. Apply the agent's action as velocity commands
        2. Step the physics engine
        3. Compute the new observation, reward, and done flags
        """
        self.current_step += 1

        forward_input = float(np.clip(action[0], -1.0, 1.0))
        turn_input = float(np.clip(action[1], -1.0, 1.0))

        linear_vel = forward_input * self.max_linear_velocity
        angular_vel = turn_input * self.max_angular_velocity

        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_forward = rot[:, 0]

        world_vel = [
            linear_vel * car_forward[0],
            linear_vel * car_forward[1],
            0.0,
        ]
        p.resetBaseVelocity(
            self.car_id, linearVelocity=world_vel,
            angularVelocity=[0.0, 0.0, angular_vel],
            physicsClientId=self.physics_client,
        )

        # Keep the car at a fixed height (top-down 2-D physics)
        fixed_z = self.config["spawn"]["position"][2]
        if abs(car_pos[2] - fixed_z) > 0.001:
            p.resetBasePositionAndOrientation(
                self.car_id, [car_pos[0], car_pos[1], fixed_z], car_orn,
                physicsClientId=self.physics_client,
            )

        p.stepSimulation(physicsClientId=self.physics_client)

        observation = self._get_observation()
        reward = self._compute_reward(action)
        self._last_step_reward = reward
        self.episode_reward += reward

        terminated = self._is_terminated()
        truncated = self._is_truncated()

        new_pos, _ = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "car_position": new_pos,
            "is_on_track": self._is_on_track(new_pos),
        }
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Render the simulation (bird's-eye view)."""
        if self.render_mode == "human":
            self._draw_debug_visuals()
            cam_dist = self.track.outer_radius * 2.5
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=0, cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0], physicsClientId=self.physics_client,
            )
            return None

        elif self.render_mode == "rgb_array":
            cam_dist = self.track.outer_radius * 2.5
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0], distance=cam_dist,
                yaw=0, pitch=-89, roll=0, upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
            _, _, rgb_img, _, _ = p.getCameraImage(
                width=640, height=480, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, physicsClientId=self.physics_client,
            )
            return np.array(rgb_img)[:, :, :3]

        return None

    def _draw_debug_visuals(self):
        """Draw HUD text and LiDAR rays in the PyBullet GUI."""
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)
        heading = self._get_car_heading(car_orn)

        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        forward_vel = car_vel[0] * rot[0, 0] + car_vel[1] * rot[1, 0]
        speed = abs(forward_vel)
        is_on_track = self._is_on_track(car_pos)
        laps = self.total_progress / (2.0 * np.pi)

        # HUD text
        r = self.track.outer_radius
        text_x, text_y, text_z = -r * 1.8, r * 1.8, 0.2
        spacing = r * 0.18

        hud_lines = [
            (f"Speed: {speed:.2f}", [1, 1, 1]),
            (f"Reward: {self._last_step_reward:.3f}", [1, 1, 0]),
            (f"Ep Reward: {self.episode_reward:.1f}", [1, 0.8, 0]),
            (f"Step: {self.current_step}/{self.max_episode_steps}", [0.8, 0.8, 0.8]),
            (f"Laps: {laps:.2f}", [0, 1, 0.5]),
            (f"On Track: {is_on_track}", [0, 1, 0] if is_on_track else [1, 0, 0]),
        ]

        for i, (text, color) in enumerate(hud_lines):
            pos = [text_x, text_y - i * spacing, text_z]
            key = f"hud_{i}"
            kwargs = {}
            if key in self._debug_items:
                kwargs["replaceItemUniqueId"] = self._debug_items[key]
            self._debug_items[key] = p.addUserDebugText(
                text, pos, textColorRGB=color, textSize=1.5, lifeTime=0,
                physicsClientId=self.physics_client, **kwargs,
            )

        # LiDAR rays
        car_2d = np.array([car_pos[0], car_pos[1]])
        line_z = 0.15
        lidar_dists = self._cast_lidar_rays(car_2d, heading)

        ray_colors = [
            [0, 0.8, 1], [0, 0.9, 0.8], [0, 1, 0.5], [0, 1, 0.2], [0, 1, 0],
            [0, 1, 0.2], [0, 1, 0.5], [0, 0.9, 0.8], [0, 0.8, 1],
        ]

        for i in range(self.LIDAR_NUM_RAYS):
            angle = heading + np.radians(self.LIDAR_ANGLES_DEG[i])
            direction = np.array([np.cos(angle), np.sin(angle)])
            end = car_2d + direction * lidar_dists[i] * self.LIDAR_MAX_RANGE
            key = f"lidar_{i}"
            kwargs = {}
            if key in self._debug_items:
                kwargs["replaceItemUniqueId"] = self._debug_items[key]
            self._debug_items[key] = p.addUserDebugLine(
                [car_pos[0], car_pos[1], line_z], [end[0], end[1], line_z],
                lineColorRGB=ray_colors[i], lineWidth=2, lifeTime=0,
                physicsClientId=self.physics_client, **kwargs,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Disconnect from the PyBullet physics server."""
        if hasattr(self, "physics_client"):
            p.disconnect(physicsClientId=self.physics_client)
