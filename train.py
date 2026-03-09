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
        "total_timesteps": 50_000,
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
