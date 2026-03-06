"""Debug utility to verify car spawn position and track boundary detection."""

import numpy as np
from environment import SelfDrivingCarEnv

print("Creating environment...")
env = SelfDrivingCarEnv(render_mode=None)

print(f"\nSpawn position: {env.spawn_pos}")

print(f"\nTrack geometry:")
print(f"  Inner points shape: {env.track.inner_points.shape}")
print(f"  Outer points shape: {env.track.outer_points.shape}")
print(f"  Inner radius (config): {env.track.inner_radius}")
print(f"  Outer radius (config): {env.track.outer_radius}")
print(f"  First 3 inner points: {env.track.inner_points[:3, :2]}")
print(f"  First 3 outer points: {env.track.outer_points[:3, :2]}")

spawn_2d = np.array(env.spawn_pos[:2])
inside_larger = env._point_in_polygon(spawn_2d, env.track.inner_points[:, :2])
inside_smaller = env._point_in_polygon(spawn_2d, env.track.outer_points[:, :2])
print(f"\nOn-track check:")
print(f"  Inside outer boundary: {inside_larger}")
print(f"  Inside inner boundary: {inside_smaller}")
print(f"  On track: {inside_larger and not inside_smaller}")
print(f"  _is_on_track(): {env._is_on_track(env.spawn_pos)}")

obs, info = env.reset()
print(f"\nAfter reset:")
print(f"  Car position: {env.spawn_pos}")
print(f"  On track: {env._is_on_track(env.spawn_pos)}")

action = np.array([0.0, 0.0])
obs, reward, terminated, truncated, info = env.step(action)
print(f"\nAfter one no-op step:")
print(f"  Reward: {reward}")
print(f"  Terminated: {terminated}")
print(f"  Car position: {info['car_position']}")
print(f"  On track: {info['is_on_track']}")

env.close()
print("\nDone.")
