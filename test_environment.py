"""
Test Script for Self-Driving Car Environment

This script helps you verify that your environment implementation is working correctly.
Run this after filling in the blanks in environment.py to test your implementation.

Questions to consider:
- Does the environment reset properly?
- Are observations in the correct format?
- Do actions work as expected?
- Are rewards being computed?
- Does termination work correctly?
"""

import numpy as np
from environment import SelfDrivingCarEnv


def test_environment_basic():
    """
    Basic environment tests.
    
    These tests verify that your environment follows the Gymnasium API correctly.
    """
    print("="*60)
    print("Testing Basic Environment Functionality")
    print("="*60)
    
    # TODO: Create environment instance
    # What render mode should you use for testing?
    # env = SelfDrivingCarEnv(config_path="setup/track_config (1).yaml", render_mode=None)
    
    try:
        # Test 1: Environment creation
        print("\n[Test 1] Creating environment...")
        # env = SelfDrivingCarEnv(config_path="setup/track_config (1).yaml", render_mode=None)
        # print("✓ Environment created successfully")
        
        # Test 2: Reset
        print("\n[Test 2] Testing reset()...")
        # obs, info = env.reset()
        # print(f"✓ Reset successful")
        # print(f"  Observation shape: {obs.shape}")
        # print(f"  Observation dtype: {obs.dtype}")
        # print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Test 3: Observation space
        print("\n[Test 3] Checking observation space...")
        # print(f"  Observation space: {env.observation_space}")
        # print(f"  Observation in space: {env.observation_space.contains(obs)}")
        # if not env.observation_space.contains(obs):
        #     print("  ✗ WARNING: Observation is not within observation space!")
        
        # Test 4: Action space
        print("\n[Test 4] Checking action space...")
        # print(f"  Action space: {env.action_space}")
        # sample_action = env.action_space.sample()
        # print(f"  Sample action: {sample_action}")
        
        # Test 5: Step
        print("\n[Test 5] Testing step()...")
        # sample_action = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(sample_action)
        # print(f"✓ Step successful")
        # print(f"  New observation shape: {obs.shape}")
        # print(f"  Reward: {reward:.4f}")
        # print(f"  Terminated: {terminated}")
        # print(f"  Truncated: {truncated}")
        
        # Test 6: Multiple steps
        print("\n[Test 6] Running multiple steps...")
        # total_reward = 0
        # for i in range(10):
        #     action = env.action_space.sample()
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     total_reward += reward
        #     if terminated or truncated:
        #         print(f"  Episode ended at step {i+1}")
        #         break
        # print(f"✓ Completed {min(10, i+1)} steps")
        # print(f"  Total reward: {total_reward:.4f}")
        
        # Test 7: Close
        print("\n[Test 7] Testing close()...")
        # env.close()
        # print("✓ Environment closed successfully")
        
        print("\n" + "="*60)
        print("All basic tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_environment_episode():
    """
    Test a full episode.
    
    This helps verify that episodes run to completion properly.
    """
    print("\n" + "="*60)
    print("Testing Full Episode")
    print("="*60)
    
    try:
        # TODO: Create environment and run a full episode
        # env = SelfDrivingCarEnv(config_path="setup/track_config (1).yaml", render_mode=None)
        # obs, info = env.reset()
        # 
        # episode_reward = 0
        # episode_length = 0
        # done = False
        # 
        # while not done:
        #     action = env.action_space.sample()
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     episode_reward += reward
        #     episode_length += 1
        #     done = terminated or truncated
        #     
        #     if episode_length > 1000:  # Safety limit
        #         print("  WARNING: Episode exceeded 1000 steps")
        #         break
        # 
        # print(f"✓ Episode completed")
        # print(f"  Episode length: {episode_length}")
        # print(f"  Episode reward: {episode_reward:.4f}")
        # 
        # env.close()
        
        print("Episode test passed! ✓")
        return True
        
    except Exception as e:
        print(f"\n✗ Episode test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_visualization():
    """
    Test environment visualization.
    
    This verifies that rendering works (if you've implemented it).
    """
    print("\n" + "="*60)
    print("Testing Visualization (Optional)")
    print("="*60)
    
    try:
        # TODO: Test rendering
        # Note: This will open a GUI window if render_mode="human"
        # Uncomment to test visualization
        # 
        # env = SelfDrivingCarEnv(config_path="setup/track_config (1).yaml", render_mode="human")
        # obs, info = env.reset()
        # 
        # print("Running 100 steps with visualization...")
        # print("(Close the PyBullet window when done)")
        # 
        # for i in range(100):
        #     action = env.action_space.sample()
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     env.render()
        #     if terminated or truncated:
        #         obs, info = env.reset()
        # 
        # env.close()
        # print("✓ Visualization test completed")
        
        print("(Skipping visualization test - uncomment code to test)")
        return True
        
    except Exception as e:
        print(f"\n✗ Visualization test failed: {e}")
        return False


def main():
    """
    Run all tests.
    """
    print("\n" + "="*60)
    print("Environment Test Suite")
    print("="*60)
    print("\nThis script tests your environment implementation.")
    print("Make sure you've filled in the TODOs in environment.py first!\n")
    
    # Run tests
    test1 = test_environment_basic()
    test2 = test_environment_episode()
    test3 = test_environment_visualization()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Basic tests: {'✓ PASSED' if test1 else '✗ FAILED'}")
    print(f"Episode test: {'✓ PASSED' if test2 else '✗ FAILED'}")
    print(f"Visualization: {'✓ PASSED' if test3 else '○ SKIPPED'}")
    
    if test1 and test2:
        print("\n🎉 Your environment is ready for training!")
        print("You can now proceed to train.py to set up your PPO agent.")
    else:
        print("\n⚠️  Some tests failed. Check your environment implementation.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

