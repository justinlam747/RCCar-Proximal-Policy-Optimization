"""
Training script for the PPO self-driving car agent.

Configures hyperparameters, sets up the training and evaluation environments,
and runs the full Stable-Baselines3 training loop with periodic checkpoints
and evaluations.
"""

import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import SelfDrivingCarEnv


def create_env(config_path, render_mode=None, log_dir=None):
    """
    Create a Monitor-wrapped, vectorized environment.

    Args:
        config_path: Path to the track/physics YAML config.
        render_mode: None for headless training, "human" to watch.
        log_dir: Directory for Monitor CSV logs (optional).

    Returns:
        A DummyVecEnv containing one environment instance.
    """
    def _make_env():
        env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
        env = Monitor(env, log_dir) if log_dir else Monitor(env)
        return env

    return DummyVecEnv([_make_env])


def load_config(config_path="setup/track_config (1).yaml"):
    """
    Build the full training configuration dictionary.

    Reads the environment YAML for reference and defines all PPO
    hyperparameters and training schedule settings.
    """
    env_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            env_config = yaml.safe_load(f)

    config = {
        # PPO hyperparameters
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,

        # Training schedule
        "total_timesteps": 200_000,
        "eval_freq": 10_000,
        "n_eval_episodes": 5,
        "checkpoint_freq": 25_000,

        # Environment
        "env_config_path": config_path,
    }
    return config


def main():
    """Set up directories, environments, model, callbacks, and run training."""

    models_dir = "models/"
    logs_dir = "logs/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    config = load_config()

    print("=" * 50)
    print("PPO Self-Driving Car -- Training")
    print("=" * 50)
    print(f"Total timesteps : {config['total_timesteps']}")
    print(f"Learning rate   : {config['learning_rate']}")
    print(f"Batch size      : {config['batch_size']}")
    print("=" * 50)

    # Environments
    train_env = create_env(config["env_config_path"], render_mode=None, log_dir=logs_dir)
    eval_env = create_env(config["env_config_path"], render_mode=None)

    # PPO agent
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
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        tensorboard_log=logs_dir,
        device="cuda",
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nModel initialized with {total_params:,} parameters")

    # Callbacks
    best_model_path = os.path.join(models_dir, "best_model")
    checkpoint_path = os.path.join(models_dir, "checkpoints")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=logs_dir,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=checkpoint_path,
        name_prefix="ppo_car",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Train
    print("\nStarting training...")
    print("View progress with: tensorboard --logdir=logs/")
    print("-" * 50)

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(models_dir, "ppo_car_final")
    model.save(final_path)

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Best model  : {best_model_path}/")
    print(f"Checkpoints : {checkpoint_path}/")
    print(f"Final model : {final_path}")
    print(f"\nTensorboard : tensorboard --logdir=logs/")


if __name__ == "__main__":
    main()
