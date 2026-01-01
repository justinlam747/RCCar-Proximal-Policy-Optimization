# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## Step 1: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `stable-baselines3[extra]` - RL algorithms & utilities
- `gymnasium` - Environment API
- `pybullet` - Physics simulation
- `torch` - Neural network backend
- `numpy`, `matplotlib`, `pyyaml` - Core utilities

---

## Step 3: Test the Environment

```bash
python test_environment.py
```

> **Expected**: All tests pass with ✓ marks. Type `n` to skip the visualization test.

---

## Step 4: Train the Agent

```bash
python train.py
```

> **Duration**: ~10-30 minutes for 200k steps

---

## Step 5: Monitor Training (Optional)

Open a new terminal while training:

```bash
tensorboard --logdir=logs/
```

Then open `http://localhost:6006` in your browser.

---

## Step 6: Evaluate Trained Model

```bash
# Without visualization (headless)
python evaluate.py --model models/best_model/best_model --episodes 10

# With visualization (opens PyBullet GUI)
python evaluate.py --model models/best_model/best_model --episodes 5 --render
```

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `python test_environment.py` | Verify environment works |
| `python train.py` | Train the PPO agent |
| `tensorboard --logdir=logs/` | View training graphs |
| `python evaluate.py --render` | Watch trained agent |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv and run `pip install -r requirements.txt` |
| PyBullet window won't open | Install display drivers; use `--render` flag |
| Training too slow | Reduce `total_timesteps` in `train.py` |
| Out of memory | Reduce `n_steps` or `batch_size` in `train.py` |
