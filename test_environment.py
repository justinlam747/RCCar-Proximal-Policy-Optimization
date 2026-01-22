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
