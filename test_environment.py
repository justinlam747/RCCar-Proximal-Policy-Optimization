"""
Test suite for verifying the SelfDrivingCarEnv implementation.

Run this before training to make sure the environment follows the
Gymnasium API and produces valid observations, rewards, and done signals.
"""

import numpy as np
from environment import SelfDrivingCarEnv

CONFIG_PATH = "setup/track_config (1).yaml"


def test_basic():
    """Verify core Gymnasium API: create, reset, step, close."""
    print("=" * 60)
    print("Testing Basic Environment Functionality")
    print("=" * 60)

    env = None
    try:
        # Creation
        print("\n[Test 1] Creating environment...")
        env = SelfDrivingCarEnv(config_path=CONFIG_PATH, render_mode=None)
        print("  PASS - Environment created")

        # Reset
        print("\n[Test 2] Testing reset()...")
        obs, info = env.reset()
        print(f"  PASS - Observation shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"         Range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Observation space
        print("\n[Test 3] Checking observation space...")
        print(f"  Space: {env.observation_space}")
        assert env.observation_space.contains(obs), "Observation outside declared space!"
        print("  PASS - Observation is within bounds")

        # Action space
        print("\n[Test 4] Checking action space...")
        print(f"  Space: {env.action_space}")
        sample = env.action_space.sample()
        print(f"  Sample action: {sample}")
        print("  PASS")

        # Single step
        print("\n[Test 5] Testing step()...")
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"  PASS - Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")

        # Multiple steps
        print("\n[Test 6] Running 10 random steps...")
        total_reward = 0.0
        steps = 0
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            steps = i + 1
            if terminated or truncated:
                print(f"  Episode ended at step {steps}")
                break
        print(f"  PASS - Completed {steps} steps, total reward: {total_reward:.4f}")

        # Close
        print("\n[Test 7] Testing close()...")
        env.close()
        env = None
        print("  PASS - Environment closed")

        print("\n" + "=" * 60)
        print("All basic tests passed.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n  FAIL - {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        return False


def test_full_episode():
    """Run a complete episode with random actions to verify termination logic."""
    print("\n" + "=" * 60)
    print("Testing Full Episode")
    print("=" * 60)

    env = None
    try:
        env = SelfDrivingCarEnv(config_path=CONFIG_PATH, render_mode=None)
        obs, info = env.reset()

        total_reward = 0.0
        length = 0
        terminated = truncated = False

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            length += 1
            if length > 1000:
                print("  WARNING: Episode exceeded 1000 steps, stopping early")
                break

        end_reason = "termination" if terminated else "truncation"
        print(f"  PASS - Length: {length}, Reward: {total_reward:.4f}, Ended by: {end_reason}")

        env.close()
        return True

    except Exception as e:
        print(f"\n  FAIL - {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        return False


def test_visualization():
    """Open a PyBullet GUI window and run 100 random steps."""
    print("\n" + "=" * 60)
    print("Testing Visualization (Optional)")
    print("=" * 60)

    env = None
    try:
        print("  Creating environment with GUI rendering...")
        env = SelfDrivingCarEnv(config_path=CONFIG_PATH, render_mode="human")
        obs, info = env.reset()

        print("  Running 100 steps with visualization...")
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            env.render()
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        print("  PASS - Visualization test completed")
        return True

    except Exception as e:
        print(f"\n  FAIL - {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        return False


def main():
    print("\n" + "=" * 60)
    print("Self-Driving Car Environment -- Test Suite")
    print("=" * 60)

    basic_ok = test_basic()
    episode_ok = test_full_episode()

    # Optional visualization test
    viz_ok = True
    print("\n" + "=" * 60)
    try:
        response = input("Run visualization test? (y/n): ").strip().lower()
        if response == "y":
            viz_ok = test_visualization()
        else:
            print("  (Skipped)")
    except EOFError:
        print("  (Skipped -- non-interactive mode)")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Basic tests   : {'PASSED' if basic_ok else 'FAILED'}")
    print(f"  Episode test  : {'PASSED' if episode_ok else 'FAILED'}")
    print(f"  Visualization : {'PASSED' if viz_ok else 'FAILED'}")

    if basic_ok and episode_ok:
        print("\nEnvironment is ready for training. Run: python train.py")
    else:
        print("\nSome tests failed -- check the errors above before training.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
