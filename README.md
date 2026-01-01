# SB3 PPO Self-Driving Car Template

This is a template project for implementing a Proximal Policy Optimization (PPO) agent using Stable-Baselines3 (SB3) for a self-driving car environment. The skeleton code is provided, and students can fill in the implementation details and experiment with hyperparameters.

## Setup Instructions

### 1. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

You can verify that everything is installed correctly by running:

```bash
python -c "import stable_baselines3; import gymnasium; print('Installation successful!')"
```

## Project Structure

```
sb3ppotemplate/
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── environment.py           # Custom self-driving car environment (to be implemented)
├── train.py                 # Training script (to be implemented)
├── evaluate.py              # Evaluation script (to be implemented)
├── test_environment.py      # Test script to verify environment works
├── setup/                   # Pre-built components
│   ├── track (3).py        # Track generation class
│   ├── controls (1).py     # Car physics controller
│   ├── track_config (1).yaml # Configuration file
│   └── car (1).urdf        # Car model file
└── models/                  # Directory for saved models
```

## Learning Approach

The code uses a **Socratic method** approach - instead of giving you the answers, it asks questions to guide your thinking:
- **Why** questions help you understand the reasoning behind design choices
- **What** questions help you identify what needs to be implemented
- **How** questions guide you through the implementation process

Each file contains:
- `TODO` comments marking places where you need to fill in code
- Socratic questions prompting you to think about design decisions
- Hints and explanations of key concepts
- Placeholder code showing the expected structure

## Next Steps

1. **Implement the environment** (`environment.py`):
   - Define observation and action spaces
   - Implement reward function
   - Set up termination/truncation conditions
   - Integrate with PyBullet and the setup components

2. **Test your environment** (`test_environment.py`):
   - Run `python test_environment.py` to verify your implementation
   - Fix any issues before proceeding to training
   - This helps catch bugs early!

3. **Set up training** (`train.py`):
   - Configure PPO hyperparameters
   - Set up callbacks for evaluation and checkpointing
   - Start training and monitor progress

4. **Create evaluation** (`evaluate.py`):
   - Load trained models
   - Run evaluation episodes
   - Visualize and analyze results

5. **Experiment**:
   - Try different hyperparameters
   - Modify reward functions
   - Adjust observation spaces
   - Compare different training configurations

## Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## Notes

- Make sure to activate your virtual environment before running any scripts
- Models will be saved in the `models/` directory
- Adjust hyperparameters in the training script to experiment with different configurations

