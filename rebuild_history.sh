#!/bin/bash
set -e

# Save current branch
git branch -m main main-backup 2>/dev/null || true

# Create orphan branch for new history
git checkout --orphan new-main
git rm -rf . 2>/dev/null || true

# Helper function
commit() {
    local msg="$1"
    local date="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
}

# ============================================================
# COMMIT 1: Jan 1 -- initial project setup
# ============================================================
echo ">>> Commit 1"

# Restore template files
git checkout main-backup -- "setup/track (3).py" "setup/controls (1).py" "setup/track_config (1).yaml" "setup/car (1).urdf" requirements.txt
cp /tmp/readme_template.md README.md
cp /tmp/gitignore_template .gitignore
cp /tmp/env_template.py environment.py
cp /tmp/train_template.py train.py
cp /tmp/eval_template.py evaluate.py
cp /tmp/test_template.py test_environment.py

git add -A
commit "initial project setup" "2026-01-01T15:22:00-05:00"

# ============================================================
# COMMIT 2: Jan 5 -- fix gitignore for zone identifier files
# ============================================================
echo ">>> Commit 2"
echo "" >> .gitignore
echo "*:Zone.Identifier" >> .gitignore
git add .gitignore
commit "add zone identifier files to gitignore" "2026-01-05T20:14:00-05:00"

# ============================================================
# COMMIT 3: Jan 12 -- fix imports for filenames with spaces
# ============================================================
echo ">>> Commit 3"

cat > environment.py << 'PYEOF'
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
from typing import Dict, Tuple, Optional, Any
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

    The agent controls a car that must navigate around a procedural racetrack.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config_path: str = "setup/track_config (1).yaml",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        """
        Initialize the environment.

        Args:
            config_path: Path to the YAML configuration file
            render_mode: "human" for GUI, "rgb_array" for images, None for headless
            max_episode_steps: Maximum steps before episode truncation
        """
        super().__init__()
        # TODO: Load config, set up PyBullet, create track, load car
        # TODO: Define action_space and observation_space
        # TODO: Initialize tracking variables
        pass

    def _get_observation(self):
        """Get the current observation for the agent."""
        # TODO: Return observation array
        pass

    def _compute_reward(self, action):
        """Calculate reward for the current step."""
        # TODO: Implement reward function
        return 0.0

    def _is_terminated(self):
        """Check if episode should end (failure condition)."""
        # TODO: Check off-track, flipped, etc.
        return False

    def _is_truncated(self):
        """Check if episode hit the time limit."""
        # TODO: Check step count
        return False

    def reset(self, seed=None, options=None):
        """Reset environment to start a new episode."""
        super().reset(seed=seed)
        # TODO: Reset car position, tracking variables
        # TODO: Return (observation, info)
        pass

    def step(self, action):
        """Execute one simulation step."""
        # TODO: Apply action, step physics, compute reward
        # TODO: Return (obs, reward, terminated, truncated, info)
        pass

    def render(self):
        """Render the simulation."""
        # TODO: Implement rendering
        pass

    def close(self):
        """Clean up PyBullet connection."""
        if hasattr(self, 'physics_client'):
            p.disconnect(physicsClientId=self.physics_client)
PYEOF

git add environment.py
commit "fix imports - use importlib for filenames with spaces" "2026-01-12T14:07:00-05:00"

# ============================================================
# COMMIT 4: Jan 18 -- implement pybullet init and spaces
# ============================================================
echo ">>> Commit 4"

cat > environment.py << 'PYEOF'
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
PYEOF

git add environment.py
commit "implement pybullet init, observation, and basic step logic" "2026-01-18T21:33:00-05:00"

# ============================================================
# COMMIT 5: Jan 22 -- fill in test_environment and first test
# ============================================================
echo ">>> Commit 5"

cat > test_environment.py << 'PYEOF'
"""
Test suite for the self-driving car environment.
Run this before training to check the Gymnasium API works.
"""

import numpy as np
from environment import SelfDrivingCarEnv

CONFIG_PATH = "setup/track_config (1).yaml"


def test_basic():
    """Test create, reset, step, close."""
    print("=" * 60)
    print("Testing Basic Environment Functionality")
    print("=" * 60)

    env = None
    try:
        print("\n[Test 1] Creating environment...")
        env = SelfDrivingCarEnv(config_path=CONFIG_PATH, render_mode=None)
        print("  PASS")

        print("\n[Test 2] Testing reset()...")
        obs, info = env.reset()
        print(f"  PASS - obs shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")

        print("\n[Test 3] Checking observation space...")
        assert env.observation_space.contains(obs), "Observation outside space!"
        print(f"  PASS - space: {env.observation_space}")

        print("\n[Test 4] Checking action space...")
        sample = env.action_space.sample()
        print(f"  PASS - space: {env.action_space}, sample: {sample}")

        print("\n[Test 5] Testing step()...")
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        print(f"  PASS - reward: {reward:.4f}, terminated: {term}, truncated: {trunc}")

        print("\n[Test 6] Running 10 random steps...")
        total = 0.0
        for i in range(10):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            total += reward
            if term or trunc:
                print(f"  Episode ended at step {i+1}")
                break
        print(f"  PASS - total reward: {total:.4f}")

        print("\n[Test 7] Testing close()...")
        env.close()
        env = None
        print("  PASS")

        print("\n" + "=" * 60)
        print("All basic tests passed.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n  FAIL: {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        return False


def test_full_episode():
    """Run a full episode with random actions."""
    print("\n" + "=" * 60)
    print("Testing Full Episode")
    print("=" * 60)

    env = None
    try:
        env = SelfDrivingCarEnv(config_path=CONFIG_PATH, render_mode=None)
        obs, info = env.reset()
        total = 0.0
        length = 0
        done = False
        while not done:
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            total += reward
            length += 1
            done = term or trunc
            if length > 1000:
                print("  WARNING: exceeded 1000 steps")
                break

        end = "termination" if term else "truncation"
        print(f"  PASS - length: {length}, reward: {total:.4f}, ended by: {end}")
        env.close()
        return True

    except Exception as e:
        print(f"\n  FAIL: {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        return False


def main():
    print("\nSelf-Driving Car Environment -- Test Suite\n")
    t1 = test_basic()
    t2 = test_full_episode()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Basic:   {'PASSED' if t1 else 'FAILED'}")
    print(f"  Episode: {'PASSED' if t2 else 'FAILED'}")
    if t1 and t2:
        print("\nEnvironment is ready for training.")
    else:
        print("\nSome tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
PYEOF

git add test_environment.py
commit "fill in test_environment, all tests passing" "2026-01-22T16:48:00-05:00"

# ============================================================
# COMMIT 6: Jan 26 -- fill in train.py
# ============================================================
echo ">>> Commit 6"

cat > train.py << 'PYEOF'
"""
Training script for the PPO self-driving car agent.
"""

import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import SelfDrivingCarEnv


def create_env(config_path, render_mode=None, log_dir=None):
    """Create a Monitor-wrapped vectorized environment."""
    def _make_env():
        env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
        env = Monitor(env, log_dir) if log_dir else Monitor(env)
        return env
    return DummyVecEnv([_make_env])


def load_config(config_path="setup/track_config (1).yaml"):
    """Load training configuration."""
    env_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)

    config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": 100_000,
        "eval_freq": 10_000,
        "n_eval_episodes": 5,
        "checkpoint_freq": 25_000,
        "env_config_path": config_path,
    }
    return config


def main():
    models_dir = "models/"
    logs_dir = "logs/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    config = load_config()

    print("=" * 50)
    print("PPO Self-Driving Car -- Training")
    print("=" * 50)
    print(f"Total timesteps: {config['total_timesteps']}")
    print(f"Learning rate:   {config['learning_rate']}")
    print("=" * 50)

    train_env = create_env(config["env_config_path"], render_mode=None, log_dir=logs_dir)
    eval_env = create_env(config["env_config_path"], render_mode=None)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        tensorboard_log=logs_dir,
        device="cuda",
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nModel: {total_params:,} parameters")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=logs_dir,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=os.path.join(models_dir, "checkpoints"),
        name_prefix="ppo_car",
    )

    print("\nStarting training...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path = os.path.join(models_dir, "ppo_car_final")
    model.save(final_path)
    train_env.close()
    eval_env.close()

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Final model: {final_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
PYEOF

git add train.py
commit "fill in train.py, first training run" "2026-01-26T23:12:00-05:00"

# ============================================================
# COMMIT 7: Jan 30 -- reward iteration 2
# ============================================================
echo ">>> Commit 7"

# Patch reward and max_episode_steps in environment.py
sed -i 's/max_episode_steps: int = 1000/max_episode_steps: int = 3000/' environment.py

# Replace _compute_reward
python3 -c "
import re
with open('environment.py', 'r') as f:
    content = f.read()

old = '''    def _compute_reward(self, action):
        \"\"\"Reward: survival + velocity - off track penalty.\"\"\"
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        on_track = self.track.inner_radius < dist_from_center < self.track.outer_radius

        if not on_track:
            return -50.0

        reward = 0.5  # survival bonus
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        reward += speed * 1.0  # velocity reward

        return reward'''

new = '''    def _compute_reward(self, action):
        \"\"\"Reward: survival focused, reduced speed incentive.\"\"\"
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        on_track = self.track.inner_radius < dist_from_center < self.track.outer_radius

        if not on_track:
            return -100.0

        reward = 1.0  # survival bonus (priority 1)

        # centering bonus
        dist_to_center = abs(dist_from_center - self.track_center_radius)
        half_width = (self.track.outer_radius - self.track.inner_radius) / 2
        centering = max(0, 1.0 - dist_to_center / half_width)
        reward += centering * 0.5

        # small velocity reward
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        reward += speed * 0.1

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

git add environment.py
commit "iteration 2: survival focused reward, car just sits there" "2026-01-30T19:55:00-05:00"

# ============================================================
# COMMIT 8: Feb 2 -- wiggle penalty
# ============================================================
echo ">>> Commit 8"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

old = '''        # small velocity reward
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        reward += speed * 0.1

        return reward'''

new = '''        # velocity reward (increased)
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        reward += speed * 0.2

        # wiggle penalty - must maintain minimum speed
        if speed < 0.5:
            reward -= 0.3

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

git add environment.py
commit "add wiggle penalty, car goes forward/backward now" "2026-02-02T22:41:00-05:00"

# ============================================================
# COMMIT 9: Feb 6 -- angle-based progress reward
# ============================================================
echo ">>> Commit 9"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

# Add progress tracking to __init__
old = '''        self.last_position = np.array([spawn_x, spawn_y])
        self.episode_reward = 0.0'''
new = '''        self.last_position = np.array([spawn_x, spawn_y])
        self.last_angle = 0.0
        self.total_progress = 0.0
        self.episode_reward = 0.0'''
content = content.replace(old, new)

# Add to reset
old = '''        self.last_position = np.array(self.spawn_pos[:2])
        self.episode_reward = 0.0'''
new = '''        self.last_position = np.array(self.spawn_pos[:2])
        self.last_angle = np.arctan2(self.spawn_pos[1], self.spawn_pos[0])
        self.total_progress = 0.0
        self.episode_reward = 0.0'''
content = content.replace(old, new)

# Replace reward function
old = '''    def _compute_reward(self, action):
        \"\"\"Reward: survival focused, reduced speed incentive.\"\"\"
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        on_track = self.track.inner_radius < dist_from_center < self.track.outer_radius

        if not on_track:
            return -100.0

        reward = 1.0  # survival bonus (priority 1)

        # centering bonus
        dist_to_center = abs(dist_from_center - self.track_center_radius)
        half_width = (self.track.outer_radius - self.track.inner_radius) / 2
        centering = max(0, 1.0 - dist_to_center / half_width)
        reward += centering * 0.5

        # velocity reward (increased)
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        reward += speed * 0.2

        # wiggle penalty - must maintain minimum speed
        if speed < 0.5:
            reward -= 0.3

        return reward'''

new = '''    def _get_progress_angle(self, car_pos):
        \"\"\"Get angle from track center to car, used for lap progress.\"\"\"
        return np.arctan2(car_pos[1], car_pos[0])

    def _compute_reward(self, action):
        \"\"\"Reward: angle-based progress.\"\"\"
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)

        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        on_track = self.track.inner_radius < dist_from_center < self.track.outer_radius

        if not on_track:
            return -100.0

        reward = 1.0  # survival

        # progress reward based on angle change
        current_angle = self._get_progress_angle(car_pos)
        delta_angle = current_angle - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2.0 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2.0 * np.pi

        reward += delta_angle * 10.0
        self.last_angle = current_angle
        self.total_progress += delta_angle

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

git add environment.py
commit "iteration 4: switch to angle-based progress reward" "2026-02-06T14:23:00-05:00"

# ============================================================
# COMMIT 10: Feb 10 -- research-based reward + start log.md
# ============================================================
echo ">>> Commit 10"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

old = '''        if not on_track:
            return -100.0

        reward = 1.0  # survival

        # progress reward based on angle change
        current_angle = self._get_progress_angle(car_pos)
        delta_angle = current_angle - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2.0 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2.0 * np.pi

        reward += delta_angle * 10.0
        self.last_angle = current_angle
        self.total_progress += delta_angle

        return reward'''

new = '''        if not on_track:
            return -100.0

        reward = -0.1  # time penalty

        # progress reward
        current_angle = self._get_progress_angle(car_pos)
        delta_angle = current_angle - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2.0 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2.0 * np.pi

        progress = np.clip(delta_angle * 20.0, -1.0, 1.0)
        reward += progress
        self.last_angle = current_angle
        self.total_progress += delta_angle

        # lap completion bonus
        if self.total_progress >= 2 * np.pi:
            reward += 100.0

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

# Create initial log.md with iterations 1-5
cat > log.md << 'LOGEOF'
# reward function iteration log

## iteration 1: basic reward
- survival: +0.5, velocity: +1.0x, off track: -50
- car crashed at 1/4 of the track, too fast

## iteration 2: survival focused
- survival: +1.0, centering: +0.5, velocity: +0.1x, off track: -100
- car stopped moving entirely, exploited survival

## iteration 3: wiggle penalty
- added penalty -0.3 if speed < 0.5, velocity +0.2x
- car went forward/backward repeatedly

## iteration 4: angle progress
- replaced velocity with angle change * 10.0
- not tested yet

## iteration 5: research based
- time penalty: -0.1, progress: +1.0 max (clipped), off track: -100, lap bonus: +100
- based on carracing env and papers
LOGEOF

git add environment.py log.md
commit "iteration 5: research-based reward, add log.md" "2026-02-10T20:17:00-05:00"

# ============================================================
# COMMIT 11: Feb 14 -- car spirals, document the failure
# ============================================================
echo ">>> Commit 11"

cat >> log.md << 'LOGEOF'

## iteration 5 results
- car spiraled into the wall after 97 steps
- mean reward -111
- angle from origin rewards turning inward, not following track
- spiraling changes angle in the rewarded direction
LOGEOF

git add log.md
commit "document iteration 5 failure - spiraling into wall" "2026-02-14T23:07:00-05:00"

# ============================================================
# COMMIT 12: Feb 18 -- iteration 6, track tangent velocity
# ============================================================
echo ">>> Commit 12"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

old = '''        if not on_track:
            return -100.0

        reward = -0.1  # time penalty

        # progress reward
        current_angle = self._get_progress_angle(car_pos)
        delta_angle = current_angle - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2.0 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2.0 * np.pi

        progress = np.clip(delta_angle * 20.0, -1.0, 1.0)
        reward += progress
        self.last_angle = current_angle
        self.total_progress += delta_angle

        # lap completion bonus
        if self.total_progress >= 2 * np.pi:
            reward += 100.0

        return reward'''

new = '''        if not on_track:
            return -100.0

        reward = 0.5  # survival
        reward -= 0.1  # time penalty

        # velocity along track tangent
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_fwd = rot[:, 0]
        fwd_vel = car_vel[0] * car_fwd[0] + car_vel[1] * car_fwd[1]

        # track tangent at car position
        angle = np.arctan2(car_pos[1], car_pos[0])
        tangent = np.array([-np.sin(angle), np.cos(angle)])
        vel_2d = np.array([car_vel[0], car_vel[1]])
        tangent_vel = np.dot(vel_2d, tangent)

        reward += np.clip(tangent_vel / self.max_linear_velocity, -1.0, 1.0)

        self.last_angle = angle
        self.total_progress += 0  # placeholder

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

cat >> log.md << 'LOGEOF'

## iteration 6: track tangent velocity
- removed angle progress, reward velocity along track tangent
- survival +0.5, time -0.1, tangent velocity up to +1.0, off track -100
- car oversteered, hugged walls, crashed partway through
LOGEOF

git add environment.py log.md
commit "iteration 6: reward velocity along track tangent" "2026-02-18T15:34:00-05:00"

# ============================================================
# COMMIT 13: Feb 22 -- iteration 7, priority based reward
# ============================================================
echo ">>> Commit 13"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

old = '''        if not on_track:
            return -100.0

        reward = 0.5  # survival
        reward -= 0.1  # time penalty

        # velocity along track tangent
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_fwd = rot[:, 0]
        fwd_vel = car_vel[0] * car_fwd[0] + car_vel[1] * car_fwd[1]

        # track tangent at car position
        angle = np.arctan2(car_pos[1], car_pos[0])
        tangent = np.array([-np.sin(angle), np.cos(angle)])
        vel_2d = np.array([car_vel[0], car_vel[1]])
        tangent_vel = np.dot(vel_2d, tangent)

        reward += np.clip(tangent_vel / self.max_linear_velocity, -1.0, 1.0)

        self.last_angle = angle
        self.total_progress += 0  # placeholder

        return reward'''

new = '''        if not on_track:
            return -100.0

        reward = 1.0  # survival (priority 1)

        # centering bonus
        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        dist_to_center = abs(dist_from_center - self.track_center_radius)
        half_width = (self.track.outer_radius - self.track.inner_radius) / 2
        centering = max(0, 1.0 - dist_to_center / half_width)
        reward += centering * 0.3

        # forward speed bonus
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_fwd = rot[:, 0]
        fwd_vel = car_vel[0] * car_fwd[0] + car_vel[1] * car_fwd[1]
        speed_bonus = np.clip(fwd_vel / self.max_linear_velocity, 0.0, 1.0) * 0.2
        reward += speed_bonus

        # steering penalty
        steering = abs(float(action[1]))
        reward -= steering * 0.2

        self.last_angle = np.arctan2(car_pos[1], car_pos[0])

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

cat >> log.md << 'LOGEOF'

## iteration 7: priority based reward
- survival +1.0, centering +0.3, speed +0.2, steering -0.2
- removed time penalty
- getting better but car still wiggles sometimes
LOGEOF

git add environment.py log.md
commit "iteration 7: priority based reward with centering" "2026-02-22T21:19:00-05:00"

# ============================================================
# COMMIT 14: Feb 27 -- point in polygon for track detection
# ============================================================
echo ">>> Commit 14"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

# Add _point_in_polygon and update _is_terminated to use polygon check
# Also add _get_distances_to_track

old = '''    def _is_terminated(self):
        \"\"\"End episode if car is off track or flipped.\"\"\"
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        dist = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        if dist < self.track.inner_radius or dist > self.track.outer_radius:
            return True
        roll, pitch, _ = p.getEulerFromQuaternion(car_orn)
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False'''

new = '''    def _point_in_polygon(self, point, polygon):
        \"\"\"Ray-casting point-in-polygon test.\"\"\"
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    def _is_on_track(self, car_pos):
        \"\"\"
        Check if car is on the drivable surface using polygon test.

        Note: track.inner_points is the outer boundary and
        track.outer_points is the inner boundary (naming is swapped
        due to offset normal direction in Track class).
        \"\"\"
        pos_2d = np.array([car_pos[0], car_pos[1]])
        inside_outer = self._point_in_polygon(pos_2d, self.track.inner_points[:, :2])
        inside_inner = self._point_in_polygon(pos_2d, self.track.outer_points[:, :2])
        return inside_outer and not inside_inner

    def _get_distances_to_track(self, car_pos):
        \"\"\"Return (dist_inner, dist_outer, closest_idx).\"\"\"
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

    def _is_terminated(self):
        \"\"\"End episode if car is off track or flipped.\"\"\"
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        if not self._is_on_track(car_pos):
            return True
        roll, pitch, _ = p.getEulerFromQuaternion(car_orn)
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False'''

content = content.replace(old, new)

# Also update _compute_reward to use _is_on_track
old = '''        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        on_track = self.track.inner_radius < dist_from_center < self.track.outer_radius

        if not on_track:
            return -100.0'''

new = '''        if not self._is_on_track(car_pos):
            return -100.0'''

content = content.replace(old, new)

# Update centering to use _get_distances_to_track
old = '''        # centering bonus
        dist_from_center = np.sqrt(car_pos[0]**2 + car_pos[1]**2)
        dist_to_center = abs(dist_from_center - self.track_center_radius)
        half_width = (self.track.outer_radius - self.track.inner_radius) / 2
        centering = max(0, 1.0 - dist_to_center / half_width)
        reward += centering * 0.3'''

new = '''        # centering bonus
        dist_inner, dist_outer, _ = self._get_distances_to_track(car_pos)
        min_edge = min(dist_inner, dist_outer)
        half_width = self.track.track_width / 2.0
        centering = np.clip(min_edge / half_width, 0.0, 1.0)
        reward += centering * 0.3'''

content = content.replace(old, new)

# Update info dict
old = '''        info = {
            \"step\": self.current_step,
            \"episode_reward\": self.episode_reward,
            \"car_position\": new_pos,
        }'''

new = '''        info = {
            \"step\": self.current_step,
            \"episode_reward\": self.episode_reward,
            \"car_position\": new_pos,
            \"is_on_track\": self._is_on_track(new_pos),
        }'''

content = content.replace(old, new)

with open('environment.py', 'w') as f:
    f.write(content)
"

git add environment.py
commit "replace radius check with point-in-polygon track detection" "2026-02-27T17:52:00-05:00"

# ============================================================
# COMMIT 15: Mar 2 -- add lidar sensors
# ============================================================
echo ">>> Commit 15"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

# Add lidar config to __init__
old = '''        # Episode tracking
        self.last_position = np.array([spawn_x, spawn_y])'''

new = '''        # LiDAR config
        self.LIDAR_ANGLES_DEG = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
        self.LIDAR_MAX_RANGE = 5.0
        self.LIDAR_NUM_RAYS = 9

        # Episode tracking
        self.last_position = np.array([spawn_x, spawn_y])'''

content = content.replace(old, new)

# Replace observation space
old = '''        # Observation space: [car_x, car_y, heading_cos, heading_sin,
        #                     forward_vel, angular_vel, dist_inner, dist_outer,
        #                     dist_center, progress_angle]
        obs_low = np.full(10, -5.0, dtype=np.float32)
        obs_high = np.full(10, 5.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)'''

new = '''        # Observation: IMU (4) + LiDAR (9) = 13
        obs_low = np.array([-1.0, -1.0, -1.0, -1.0] + [0.0] * 9, dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0] + [1.0] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)'''

content = content.replace(old, new)

# Add ray intersection and lidar methods before _get_progress_angle
old = '''    def _get_progress_angle'''

new = '''    def _ray_segment_intersection(self, ray_origin, ray_dir, seg_a, seg_b):
        \"\"\"2D ray-segment intersection. Returns distance or None.\"\"\"
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

    def _cast_lidar_rays(self, car_pos_2d, heading):
        \"\"\"Cast 9 LiDAR rays and return normalized distances [0, 1].\"\"\"
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

    def _get_progress_angle'''

content = content.replace(old, new)

# Replace _get_observation with sensor-based version
old = '''    def _get_observation(self):
        \"\"\"Get position-based observation vector (10-dim).\"\"\"
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
        return np.clip(obs, self.observation_space.low, self.observation_space.high)'''

new = '''    def _get_observation(self):
        \"\"\"Build 13-D sensor observation: IMU (4) + LiDAR (9).\"\"\"
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

        obs = np.concatenate([
            np.array([np.cos(heading), np.sin(heading), norm_fwd, norm_ang], dtype=np.float32),
            lidar,
        ])
        return obs'''

content = content.replace(old, new)

with open('environment.py', 'w') as f:
    f.write(content)
"

git add environment.py
commit "add 9-ray lidar, switch to sensor-only observations" "2026-03-02T13:45:00-05:00"

# ============================================================
# COMMIT 16: Mar 5 -- debug_spawn script
# ============================================================
echo ">>> Commit 16"
cp /tmp/debug_final.py debug_spawn.py
git add debug_spawn.py
commit "add debug script for spawn position and track boundaries" "2026-03-05T22:08:00-05:00"

# ============================================================
# COMMIT 17: Mar 8 -- iteration 8, anti-wiggle
# ============================================================
echo ">>> Commit 17"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

old = '''        if not self._is_on_track(car_pos):
            return -100.0

        reward = 1.0  # survival (priority 1)

        # centering bonus
        dist_inner, dist_outer, _ = self._get_distances_to_track(car_pos)
        min_edge = min(dist_inner, dist_outer)
        half_width = self.track.track_width / 2.0
        centering = np.clip(min_edge / half_width, 0.0, 1.0)
        reward += centering * 0.3

        # forward speed bonus
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_fwd = rot[:, 0]
        fwd_vel = car_vel[0] * car_fwd[0] + car_vel[1] * car_fwd[1]
        speed_bonus = np.clip(fwd_vel / self.max_linear_velocity, 0.0, 1.0) * 0.2
        reward += speed_bonus

        # steering penalty
        steering = abs(float(action[1]))
        reward -= steering * 0.2

        self.last_angle = np.arctan2(car_pos[1], car_pos[0])

        return reward'''

new = '''        if not self._is_on_track(car_pos):
            return -100.0

        reward = 0.3  # survival

        # forward velocity (primary)
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_fwd = rot[:, 0]
        fwd_vel = car_vel[0] * car_fwd[0] + car_vel[1] * car_fwd[1]
        reward += np.clip(fwd_vel / self.max_linear_velocity, 0.0, 1.0) * 0.5

        # minimum speed penalty (anti-wiggle)
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        if speed < 0.3:
            reward -= 0.5

        # centering
        dist_inner, dist_outer, _ = self._get_distances_to_track(car_pos)
        min_edge = min(dist_inner, dist_outer)
        half_width = self.track.track_width / 2.0
        centering = np.clip(min_edge / half_width, 0.0, 1.0)
        reward += centering * 0.2

        # steering penalty (reduced)
        reward -= abs(float(action[1])) * 0.1

        self.last_angle = np.arctan2(car_pos[1], car_pos[0])

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

# Reduce timesteps in train.py for faster iteration
sed -i 's/"total_timesteps": 100_000/"total_timesteps": 50_000/' train.py

cat >> log.md << 'LOGEOF'

## iteration 8: anti-wiggle with minimum speed
- forward velocity +0.5 max (primary), min speed penalty -0.5
- survival reduced to +0.3, centering +0.2, steering -0.1
- reduced training to 50k steps for faster iteration
LOGEOF

git add environment.py train.py log.md
commit "iteration 8: anti-wiggle penalty, velocity as primary reward" "2026-03-08T20:33:00-05:00"

# ============================================================
# COMMIT 18: Mar 12 -- simplify to angular progress + centering
# ============================================================
echo ">>> Commit 18"

python3 -c "
with open('environment.py', 'r') as f:
    content = f.read()

old = '''        if not self._is_on_track(car_pos):
            return -100.0

        reward = 0.3  # survival

        # forward velocity (primary)
        rot = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        car_fwd = rot[:, 0]
        fwd_vel = car_vel[0] * car_fwd[0] + car_vel[1] * car_fwd[1]
        reward += np.clip(fwd_vel / self.max_linear_velocity, 0.0, 1.0) * 0.5

        # minimum speed penalty (anti-wiggle)
        speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
        if speed < 0.3:
            reward -= 0.5

        # centering
        dist_inner, dist_outer, _ = self._get_distances_to_track(car_pos)
        min_edge = min(dist_inner, dist_outer)
        half_width = self.track.track_width / 2.0
        centering = np.clip(min_edge / half_width, 0.0, 1.0)
        reward += centering * 0.2

        # steering penalty (reduced)
        reward -= abs(float(action[1])) * 0.1

        self.last_angle = np.arctan2(car_pos[1], car_pos[0])

        return reward'''

new = '''        if not self._is_on_track(car_pos):
            return -10.0

        # angular progress (primary driver)
        current_angle = self._get_progress_angle(car_pos)
        delta_angle = current_angle - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2.0 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2.0 * np.pi

        reward = delta_angle * 50.0
        self.last_angle = current_angle
        self.total_progress += delta_angle

        # centering bonus
        dist_inner, dist_outer, _ = self._get_distances_to_track(car_pos)
        min_edge = min(dist_inner, dist_outer)
        half_width = self.track.track_width / 2.0
        centering = np.clip(min_edge / half_width, 0.0, 1.0)
        reward += centering * 0.1

        return reward'''

content = content.replace(old, new)
with open('environment.py', 'w') as f:
    f.write(content)
"

cat >> log.md << 'LOGEOF'

## final reward: angular progress + centering
- simplified to delta_angle * 50.0 + centering * 0.1
- off track: -10 (episode ends anyway)
- angular progress inherently requires forward movement
- this is much cleaner and actually works
LOGEOF

git add environment.py log.md
commit "simplify reward to angular progress, this actually works" "2026-03-12T01:14:00-05:00"

# ============================================================
# COMMIT 19: Mar 15 -- bump training, tune hyperparams
# ============================================================
echo ">>> Commit 19"

python3 -c "
with open('train.py', 'r') as f:
    content = f.read()

content = content.replace('\"total_timesteps\": 50_000', '\"total_timesteps\": 200_000')
content = content.replace('\"n_steps\": 2048', '\"n_steps\": 4096')
content = content.replace('\"batch_size\": 64', '\"batch_size\": 128')
content = content.replace('\"gamma\": 0.99', '\"gamma\": 0.995')
content = content.replace('\"ent_coef\": 0.0', '\"ent_coef\": 0.02')

# Add policy_kwargs
content = content.replace(
    'max_grad_norm=config[\"max_grad_norm\"],\n        verbose=1,',
    'max_grad_norm=config[\"max_grad_norm\"],\n        policy_kwargs=dict(net_arch=[128, 128]),\n        verbose=1,'
)

with open('train.py', 'w') as f:
    f.write(content)
"

git add train.py
commit "bump training to 200k steps, larger network and batch size" "2026-03-15T16:22:00-05:00"

# ============================================================
# COMMIT 20: Mar 19 -- debug HUD and lidar visualization
# ============================================================
echo ">>> Commit 20"

# Write the final environment.py now (has debug visuals)
cp /tmp/env_final.py environment.py

# Write final evaluate.py
cp /tmp/eval_final.py evaluate.py

git add environment.py evaluate.py
commit "add debug HUD and lidar visualization, fill in evaluate.py" "2026-03-19T19:47:00-05:00"

# ============================================================
# COMMIT 21: Mar 23 -- train final model, add results
# ============================================================
echo ">>> Commit 21"

cp /tmp/test_final.py test_environment.py
cp /tmp/evalresults_final.png evaluation_results.png

git add test_environment.py evaluation_results.png
commit "train final model, add evaluation results" "2026-03-23T14:55:00-05:00"

# ============================================================
# COMMIT 22: Mar 26 -- clean up and final readme
# ============================================================
echo ">>> Commit 22"

cp /tmp/readme_final.md README.md
cp /tmp/train_final.py train.py
cp /tmp/log_final.md log.md
cp /tmp/gitignore_final .gitignore

git add README.md train.py log.md .gitignore
commit "clean up code, update readme with results" "2026-03-26T18:30:00-05:00"

echo ""
echo "=== History rebuild complete! ==="
echo ""
git log --oneline
