# RC Car Proximal Policy Optimization

A reinforcement learning project that trains a self-driving RC car using Proximal Policy Optimization (PPO) in a PyBullet simulation. The car learns to navigate a procedurally generated racetrack using only onboard sensor data (IMU and LiDAR).

## Overview

The agent receives a 13-dimensional observation vector from simulated sensors and outputs continuous throttle and steering commands. Training uses Stable-Baselines3's PPO implementation with periodic evaluation and model checkpointing.

**Sensor inputs:**
- IMU: heading (cos/sin), forward velocity, angular velocity
- LiDAR: 9 distance rays spanning the front 180 degrees

**Action outputs:**
- Throttle: continuous [-1, 1] (backward to forward)
- Steering: continuous [-1, 1] (left to right)

## Project Structure

```
.
├── environment.py          # Custom Gymnasium environment (PyBullet + sensors)
├── train.py                # PPO training script with callbacks
├── evaluate.py             # Model evaluation and results visualization
├── test_environment.py     # Environment verification tests
├── debug_spawn.py          # Spawn position and track boundary debugger
├── requirements.txt        # Python dependencies
├── log.md                  # Reward function iteration history
├── setup/
│   ├── track (3).py        # Procedural track generation
│   ├── controls (1).py     # Tank-drive car physics controller
│   ├── track_config (1).yaml  # Track, physics, and camera config
│   └── car (1).urdf        # Car model for PyBullet
└── models/                 # Saved models (generated during training)
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import stable_baselines3; import gymnasium; import pybullet; print('OK')"
```

## Usage

### Test the environment

```bash
python test_environment.py
```

### Train the agent

```bash
python train.py
```

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir=logs/
```

### Evaluate a trained model

```bash
python evaluate.py --model models/best_model/best_model --episodes 10
```

Add `--render` to open a PyBullet GUI window during evaluation.

## Reward Design

The reward function went through 8 iterations (documented in `log.md`). The final version uses angular progress around the track as the primary signal, with a small centering bonus to keep the car away from walls and a penalty for leaving the track.

| Component         | Value             |
|-------------------|-------------------|
| Angular progress  | +50 * delta_angle |
| Centering bonus   | up to +0.1        |
| Off-track penalty | -10               |

## Hyperparameters

| Parameter       | Value  |
|-----------------|--------|
| Learning rate   | 3e-4   |
| Rollout steps   | 4096   |
| Batch size      | 128    |
| Epochs per update | 10   |
| Gamma           | 0.995  |
| GAE lambda      | 0.95   |
| Clip range      | 0.2    |
| Entropy coeff   | 0.02   |
| Network         | MLP [128, 128] |
| Total timesteps | 200,000 |

## Results

After training, the best model achieves a consistent reward of ~2847 per episode, running the full 3000 steps without leaving the track.

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
