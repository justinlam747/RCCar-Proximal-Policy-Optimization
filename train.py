"""
Training Script for SB3 PPO Self-Driving Car Agent

This script sets up and trains a PPO agent using Stable-Baselines3.
All hyperparameters and training configuration are now filled in.

Key Concepts:
- Learning rate controls how fast the agent updates its policy
- Batch size determines how many samples are used per gradient update
- n_steps is how many steps to collect before each policy update
- gamma (discount factor) determines how much future rewards matter
"""

import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import SelfDrivingCarEnv


def create_env(config_path: str, render_mode: str = None, log_dir: str = None):
    """
    Create and wrap the environment.
    
    Why we wrap environments:
    - Monitor: Logs episode rewards, lengths, and times for tracking progress
    - DummyVecEnv: SB3 requires vectorized environments (even for single env)
    - SubprocVecEnv: Would run multiple envs in parallel (faster but more overhead)
    
    Args:
        config_path: Path to environment config file
        render_mode: Rendering mode (None for training, "human" for visualization)
        log_dir: Directory for Monitor logs
        
    Returns:
        env: Wrapped vectorized environment
    """
    def _make_env():
        env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
        # Wrap with Monitor to log episode statistics
        # This is essential for tracking training progress!
        if log_dir is not None:
            env = Monitor(env, log_dir)
        else:
            env = Monitor(env)
        return env
    
    # DummyVecEnv runs environments sequentially (good for single env)
    # For multiple parallel envs, use SubprocVecEnv instead
    env = DummyVecEnv([_make_env])
    
    return env


def load_config(config_path: str = "setup/track_config (1).yaml"):
    """
    Load training configuration with PPO hyperparameters.
    
    These hyperparameters are tuned for the self-driving car task.
    You can modify them to experiment with different settings.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary with all hyperparameters
    """
    # Load environment config if it exists
    env_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
    
    config = {
        # ============ PPO HYPERPARAMETERS ============
        
        # Learning rate: How fast the agent learns
        # Too high = unstable, too low = slow learning
        "learning_rate": 3e-4,
        
        # Steps per update: How many steps to collect before updating the policy
        # Higher = more stable but slower, Lower = faster but noisier
        "n_steps": 2048,
        
        # Batch size: Samples per gradient update (must divide n_steps evenly)
        # Larger = more stable gradients, Smaller = faster updates
        "batch_size": 64,
        
        # Epochs per update: How many times to reuse collected data
        # Higher = more sample efficient, but can lead to overfitting
        "n_epochs": 10,
        
        # Discount factor (gamma): How much future rewards matter
        # Close to 1 = long-term thinking, Close to 0 = short-term focus
        "gamma": 0.99,
        
        # GAE lambda: Controls bias-variance tradeoff in advantage estimation
        # Higher = lower bias but higher variance
        "gae_lambda": 0.95,
        
        # Clip range: PPO's key innovation - limits how much policy can change
        # Prevents destructively large policy updates
        "clip_range": 0.2,
        
        # Entropy coefficient: Encourages exploration
        # Higher = more random exploration, Lower = exploit known strategies
        "ent_coef": 0.01,
        
        # Value function coefficient: Weight of value loss in total loss
        "vf_coef": 0.5,
        
        # Max gradient norm: Clips gradients to prevent explosion
        "max_grad_norm": 0.5,
        
        # ============ TRAINING SETTINGS ============
        
        # Total timesteps: How long to train
        # Start small and increase if learning is promising
        "total_timesteps": 200_000,
        
        # Evaluation frequency: Steps between evaluations
        "eval_freq": 10_000,
        
        # Number of evaluation episodes
        "n_eval_episodes": 5,
        
        # Checkpoint frequency: Steps between saving checkpoints
        "checkpoint_freq": 25_000,
        
        # Environment config path
        "env_config_path": config_path,
    }
    
    return config


def main():
    """
    Main training function.
    
    This sets up the complete training pipeline:
    1. Create directories for logs and models
    2. Set up training and evaluation environments
    3. Initialize PPO agent with hyperparameters
    4. Configure callbacks for monitoring and saving
    5. Train the agent
    6. Save the final model
    """
    
    # ============ SET UP DIRECTORIES ============
    models_dir = "models/"
    logs_dir = "logs/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # ============ LOAD CONFIGURATION ============
    config = load_config()
    print("=" * 50)
    print("PPO Self-Driving Car Training")
    print("=" * 50)
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 50)
    
    # ============ CREATE ENVIRONMENTS ============
    # Training environment: No rendering for speed
    train_env = create_env(
        config_path=config["env_config_path"],
        render_mode=None,  # No rendering during training for speed
        log_dir=logs_dir
    )
    
    # Evaluation environment: Also no rendering (set to "human" to watch)
    eval_env = create_env(
        config_path=config["env_config_path"],
        render_mode=None,  # Change to "human" to visualize evaluation
        log_dir=None
    )
    
    # ============ INITIALIZE PPO AGENT ============
    model = PPO(
        policy="MlpPolicy",  # Multi-layer perceptron (neural network)
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
        verbose=1,  # 1 = print training info, 0 = silent, 2 = debug
        tensorboard_log=logs_dir,  # View with: tensorboard --logdir=logs/
    )
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.policy.parameters()):,} parameters")
    
    # ============ SET UP CALLBACKS ============
    callbacks = []
    
    # Evaluation callback: Tests agent periodically and saves the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=logs_dir,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,  # Use deterministic actions for consistent evaluation
        render=False,  # Set to True to watch evaluation episodes
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback: Saves model at regular intervals (insurance against crashes)
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=os.path.join(models_dir, "checkpoints"),
        name_prefix="ppo_car",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)
    
    # ============ TRAIN THE MODEL ============
    print("\nStarting training...")
    print("View training progress with: tensorboard --logdir=logs/")
    print("-" * 50)
    
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,  # Shows a nice progress bar
    )
    
    # ============ SAVE FINAL MODEL ============
    final_model_path = os.path.join(models_dir, "ppo_car_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # ============ CLEAN UP ============
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Best model saved to: {models_dir}best_model/")
    print(f"Checkpoints saved to: {models_dir}checkpoints/")
    print(f"Final model saved to: {final_model_path}")
    print("\nTo view training logs: tensorboard --logdir=logs/")
    print("To test the model, create a test script that loads the model with PPO.load()")


if __name__ == "__main__":
    main()
