"""
Evaluation Script for SB3 PPO Self-Driving Car Agent

This script loads a trained model and evaluates its performance.
Students should implement metrics and visualization.

Questions to consider:
- How do you measure agent performance?
- What metrics are most important?
- How do you visualize agent behavior?
- Should evaluation be deterministic or stochastic?
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
    """
    Create evaluation environment.
    
    Questions:
    - Should evaluation use GUI rendering?
    - Do you need multiple parallel environments for evaluation?
    
    Args:
        config_path: Path to environment config file
        render_mode: Rendering mode ("human" for GUI, "rgb_array" for video)
        
    Returns:
        env: Environment for evaluation
    """
    def _make_env():
        env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
        # TODO: Wrap with Monitor for episode statistics
        # env = Monitor(env, ???)
        return env
    
    # TODO: Create vectorized environment
    # Do you need vectorization for evaluation?
    # env = DummyVecEnv([_make_env])
    
    return None  # Placeholder


def load_model(model_path: str, env):
    """
    Load a trained PPO model.
    
    Questions:
    - How do you load a saved model?
    - What if the model file doesn't exist?
    
    Args:
        model_path: Path to saved model
        env: Environment (needed for model loading)
        
    Returns:
        model: Loaded PPO model
    """
    # TODO: Check if model exists
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model not found: {model_path}")
    
    # TODO: Load model
    # How do you load a PPO model?
    # model = PPO.load(???, env=???)
    
    return None  # Placeholder


def evaluate_episode(model, env, render: bool = True):
    """
    Run a single evaluation episode.
    
    Questions:
    - Should actions be deterministic or stochastic?
    - What information should you collect during an episode?
    - How do you know when an episode ends?
    
    Args:
        model: Trained PPO model
        env: Environment
        render: Whether to render the episode
        
    Returns:
        episode_info: Dictionary with episode statistics
    """
    # TODO: Reset environment
    # obs, info = env.reset()
    
    # TODO: Initialize tracking variables
    # episode_reward = 0.0
    # episode_length = 0
    # done = False
    # actions_taken = []
    # rewards_received = []
    
    # TODO: Run episode
    # while not done:
    #     # Get action from model
    #     # What's the difference between predict() and sample()?
    #     # Should you use deterministic=True?
    #     # action, _ = model.predict(obs, deterministic=???)
    #     
    #     # Step environment
    #     # obs, reward, terminated, truncated, info = env.step(action)
    #     # done = terminated or truncated
    #     
    #     # Update tracking
    #     # episode_reward += reward
    #     # episode_length += 1
    #     # actions_taken.append(action)
    #     # rewards_received.append(reward)
    #     
    #     # Render if requested
    #     # if render:
    #     #     env.render()
    
    # TODO: Compile episode statistics
    # episode_info = {
    #     "reward": episode_reward,
    #     "length": episode_length,
    #     "actions": np.array(actions_taken),
    #     "rewards": np.array(rewards_received),
    # }
    
    return {}  # Placeholder


def evaluate_multiple_episodes(model, env, n_episodes: int = 10, render: bool = False):
    """
    Evaluate model over multiple episodes.
    
    Questions:
    - How many episodes give a reliable performance estimate?
    - What statistics should you compute? (mean, std, min, max?)
    - Should you render all episodes or just some?
    
    Args:
        model: Trained PPO model
        env: Environment
        n_episodes: Number of episodes to run
        render: Whether to render episodes
        
    Returns:
        results: Dictionary with evaluation statistics
    """
    # TODO: Run multiple episodes
    # episode_rewards = []
    # episode_lengths = []
    # 
    # for episode in range(n_episodes):
    #     print(f"Running episode {episode + 1}/{n_episodes}...")
    #     should_render = render and (episode == 0 or episode == n_episodes - 1)
    #     episode_info = evaluate_episode(model, env, render=should_render)
    #     
    #     episode_rewards.append(episode_info["reward"])
    #     episode_lengths.append(episode_info["length"])
    
    # TODO: Compute statistics
    # What statistics are most informative?
    # results = {
    #     "mean_reward": np.mean(episode_rewards),
    #     "std_reward": np.std(episode_rewards),
    #     "min_reward": np.min(episode_rewards),
    #     "max_reward": np.max(episode_rewards),
    #     "mean_length": np.mean(episode_lengths),
    #     "std_length": np.std(episode_lengths),
    #     "episode_rewards": episode_rewards,
    #     "episode_lengths": episode_lengths,
    # }
    
    return {}  # Placeholder


def visualize_results(results: dict, save_path: str = None):
    """
    Visualize evaluation results.
    
    Questions:
    - What plots are most informative?
    - How do you visualize reward distributions?
    - Should you plot episode-by-episode or summary statistics?
    
    Args:
        results: Evaluation results dictionary
        save_path: Optional path to save figure
    """
    # TODO: Create visualizations
    # What plots would help understand agent performance?
    # 
    # Example: Reward distribution
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 
    # # Plot 1: Reward distribution
    # axes[0].hist(results["episode_rewards"], bins=20, edgecolor="black")
    # axes[0].axvline(results["mean_reward"], color="red", linestyle="--", label="Mean")
    # axes[0].set_xlabel("Episode Reward")
    # axes[0].set_ylabel("Frequency")
    # axes[0].set_title("Reward Distribution")
    # axes[0].legend()
    # 
    # # Plot 2: Episode lengths
    # axes[1].plot(results["episode_lengths"], marker="o")
    # axes[1].axhline(results["mean_length"], color="red", linestyle="--", label="Mean")
    # axes[1].set_xlabel("Episode")
    # axes[1].set_ylabel("Episode Length")
    # axes[1].set_title("Episode Lengths")
    # axes[1].legend()
    # 
    # plt.tight_layout()
    # 
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Figure saved to {save_path}")
    # else:
    #     plt.show()
    
    pass


def print_results(results: dict):
    """
    Print evaluation results in a readable format.
    
    Questions:
    - What information is most important to display?
    - How should you format the output?
    
    Args:
        results: Evaluation results dictionary
    """
    # TODO: Print formatted results
    # print("\n" + "="*50)
    # print("EVALUATION RESULTS")
    # print("="*50)
    # print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    # print(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    # print(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    # print("="*50 + "\n")
    
    pass


def main():
    """
    Main evaluation function.
    
    Questions:
    - How should users specify which model to evaluate?
    - Should evaluation be interactive or automated?
    - What output formats are useful?
    """
    # TODO: Parse command line arguments
    # How should users specify model path, number of episodes, etc.?
    # parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    # parser.add_argument("--model", type=str, required=True, help="Path to model file")
    # parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    # parser.add_argument("--render", action="store_true", help="Render episodes")
    # parser.add_argument("--config", type=str, default="setup/track_config (1).yaml", help="Config file path")
    # args = parser.parse_args()
    
    # For now, use default values
    model_path = "models/best_model.zip"  # TODO: Get from args
    n_episodes = 10  # TODO: Get from args
    render = True  # TODO: Get from args
    config_path = "setup/track_config (1).yaml"  # TODO: Get from args
    
    # TODO: Create environment
    # What render mode should you use?
    # env = create_env(config_path, render_mode="human" if render else None)
    
    # TODO: Load model
    # model = load_model(model_path, env)
    
    # TODO: Run evaluation
    # print(f"Evaluating model: {model_path}")
    # print(f"Running {n_episodes} episodes...")
    # results = evaluate_multiple_episodes(model, env, n_episodes=n_episodes, render=render)
    
    # TODO: Display results
    # print_results(results)
    
    # TODO: Visualize results
    # visualize_results(results, save_path="evaluation_results.png")
    
    # TODO: Clean up
    # env.close()
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()

