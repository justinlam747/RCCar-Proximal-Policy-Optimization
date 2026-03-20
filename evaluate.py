"""
Evaluation script for the trained PPO self-driving car agent.

Loads a saved model, runs evaluation episodes, prints statistics, and
optionally saves a results plot.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import SelfDrivingCarEnv


def create_env(config_path: str, render_mode: str = "human"):
    """Create a Monitor-wrapped vectorized environment for evaluation."""
    def _make_env():
        env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
        return Monitor(env)

    return DummyVecEnv([_make_env])


def load_model(model_path: str, env):
    """Load a trained PPO model from disk."""
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return PPO.load(model_path, env=env)


def evaluate_episode(model, env, render: bool = True):
    """
    Run a single deterministic evaluation episode.

    Returns a dict with reward, length, action history, and per-step rewards.
    """
    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False
    actions = []
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        episode_length += 1
        actions.append(action[0])
        rewards.append(reward[0])
        if render:
            env.render()

    return {
        "reward": episode_reward,
        "length": episode_length,
        "actions": np.array(actions),
        "rewards": np.array(rewards),
    }


def evaluate_multiple_episodes(model, env, n_episodes: int = 10, render: bool = False):
    """
    Run multiple evaluation episodes and compute aggregate statistics.
    """
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        should_render = render and (ep == 0 or ep == n_episodes - 1)
        info = evaluate_episode(model, env, render=should_render)
        episode_rewards.append(info["reward"])
        episode_lengths.append(info["length"])
        print(f"  Episode {ep + 1}/{n_episodes}  --  "
              f"Reward: {info['reward']:.2f}, Length: {info['length']}")

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def visualize_results(results: dict, save_path: str = None):
    """Plot reward distribution and episode lengths."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rewards = results["episode_rewards"]
    reward_range = max(rewards) - min(rewards)
    n_bins = max(1, min(20, int(len(rewards) / 2))) if reward_range > 0 else 1

    axes[0].hist(rewards, bins=n_bins, edgecolor="black", alpha=0.7)
    axes[0].axvline(results["mean_reward"], color="red", linestyle="--",
                    label=f"Mean: {results['mean_reward']:.2f}")
    axes[0].set_xlabel("Episode Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Reward Distribution")
    axes[0].legend()

    axes[1].plot(results["episode_lengths"], marker="o", linewidth=2, markersize=6)
    axes[1].axhline(results["mean_length"], color="red", linestyle="--",
                    label=f"Mean: {results['mean_length']:.2f}")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Lengths")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def print_results(results: dict):
    """Print evaluation statistics."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Reward    : {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Reward Range   : [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print(f"Mean Ep Length : {results['mean_length']:.2f} +/- {results['std_length']:.2f}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO driving model")
    parser.add_argument("--model", type=str, default="models/best_model/best_model",
                        help="Path to the saved model file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the first and last episodes")
    parser.add_argument("--config", type=str, default="setup/track_config (1).yaml",
                        help="Path to environment config")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Path to save the results plot")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = create_env(args.config, render_mode=render_mode)

    print(f"Loading model: {args.model}")
    model = load_model(args.model, env)

    print(f"Running {args.episodes} evaluation episodes...")
    results = evaluate_multiple_episodes(model, env, n_episodes=args.episodes, render=args.render)

    print_results(results)

    save_path = args.save_plot or "evaluation_results.png"
    visualize_results(results, save_path=save_path)

    env.close()
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
