"""
Self-Driving Car Environment for SB3 PPO Training

This is a Gymnasium-compatible environment that uses PyBullet for physics simulation.
The environment integrates with the Track and TankDriveController classes from the setup folder.

TODO: Fill in the blanks marked with # TODO comments and answer the Socratic questions!
"""

import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

# Import the track and controller from setup
from setup.track import Track
from setup.controls import TankDriveController


class SelfDrivingCarEnv(gym.Env):
    """
    A self-driving car environment using PyBullet physics.
    
    The agent controls a car that must navigate around a procedural racetrack.
    Think about: What makes a good observation space for a self-driving car?
    What actions should the agent be able to take?
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config_path: str = "setup/track_config (1).yaml",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        """
        Initialize the environment.
        
        Questions to consider:
        - What information does the agent need to see? (observation space)
        - What actions can the agent take? (action space)
        - How do we define success or failure? (reward function)
        - When should an episode end? (termination conditions)
        
        Args:
            config_path: Path to the YAML configuration file
            render_mode: "human" for GUI, "rgb_array" for images, None for headless
            max_episode_steps: Maximum steps before truncation
        """
        super().__init__()
        
        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Load configuration
        # TODO: Load the YAML config file here. What information do you need from it?
        # Hint: You'll need physics settings, spawn position, etc.
        # self.config = ???
        
        # Initialize PyBullet
        # TODO: How do you connect to PyBullet? What's the difference between GUI and DIRECT mode?
        # Think about: When would you use each mode?
        if render_mode == "human":
            # ??? = p.connect(p.GUI)
            pass
        else:
            # ??? = p.connect(p.DIRECT)
            pass
        
        # self.physics_client = ???
        
        # TODO: Set up physics parameters from config
        # What physics settings affect the simulation? (gravity, time step, etc.)
        # p.setGravity(???, ???, ???, physicsClientId=self.physics_client)
        # p.setTimeStep(???, physicsClientId=self.physics_client)
        
        # Initialize track
        # TODO: Create a Track instance. What parameters does it need?
        # self.track = Track(???, ???)
        # self.track.spawn_in_pybullet(???)
        
        # Load car URDF
        # TODO: Load the car model. Where is the URDF file located?
        # Think about: What happens if the file doesn't exist?
        # car_urdf_path = ???
        # self.car_id = p.loadURDF(???, ???, physicsClientId=self.physics_client)
        
        # TODO: Set initial car position and orientation from config
        # What's the difference between position and orientation?
        # spawn_pos = ???
        # spawn_orn = ???
        # p.resetBasePositionAndOrientation(???, ???, ???, physicsClientId=self.physics_client)
        
        # Initialize controller
        # TODO: Create the TankDriveController. What does it need?
        # self.controller = TankDriveController(???, ???, ???)
        
        # Define action space
        # TODO: What actions should the agent control?
        # Think about: The TankDriveController uses forward_input (-1 to +1) and turn_input (-1 to +1)
        # What type of action space is appropriate? (Discrete, Box, etc.)
        # Should actions be continuous or discrete? Why?
        # self.action_space = spaces.???(???)
        
        # Define observation space
        # TODO: What should the agent observe?
        # Consider: Car position, velocity, orientation, distance to track boundaries, etc.
        # What shape should observations have? What are reasonable bounds?
        # self.observation_space = spaces.???(???)
        
        # TODO: Initialize any tracking variables you need
        # What information do you need to compute rewards and check termination?
        # self.last_position = ???
        # self.progress = ???
        # self.episode_reward = ???
        
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Questions to consider:
        - What information helps the agent make good decisions?
        - Should observations be normalized? Why or why not?
        - How can you extract useful features from the car's state?
        
        Returns:
            observation: The current state observation
        """
        # TODO: Get car's current state from PyBullet
        # What information can you get about the car's position, orientation, and velocity?
        # car_pos, car_orn = p.getBasePositionAndOrientation(???, physicsClientId=self.physics_client)
        # car_vel, car_ang_vel = p.getBaseVelocity(???, physicsClientId=self.physics_client)
        
        # TODO: Extract useful features
        # Think about:
        # - How can you represent orientation? (Euler angles, quaternion, rotation matrix?)
        # - What velocity components matter? (forward speed, angular velocity?)
        # - How can you measure distance to track boundaries?
        # - Should you include relative position on the track?
        
        # TODO: Combine features into observation vector
        # observation = np.array([???])
        
        # TODO: Normalize observations if needed
        # Why might normalization help training?
        
        return np.zeros(10)  # Placeholder - replace with actual observation
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute the reward for the current step.
        
        This is a critical function! The reward signal shapes what the agent learns.
        
        Questions to consider:
        - What behaviors do you want to encourage? (staying on track, making progress, etc.)
        - What behaviors should be penalized? (going off track, crashing, etc.)
        - How do you balance different objectives? (progress vs. safety)
        - Should rewards be sparse (only at milestones) or dense (every step)?
        
        Args:
            action: The action taken by the agent
            
        Returns:
            reward: The reward value
        """
        # TODO: Get current car state
        # car_pos, car_orn = ???
        
        # TODO: Check if car is on track
        # How can you determine if the car is within track boundaries?
        # Hint: Use self.track.inner_points and self.track.outer_points
        # is_on_track = ???
        
        # TODO: Calculate progress
        # How do you measure progress around the track?
        # Think about: Distance traveled, laps completed, distance from start, etc.
        # progress_reward = ???
        
        # TODO: Calculate speed reward
        # Should the agent be rewarded for speed? Why or why not?
        # speed_reward = ???
        
        # TODO: Calculate penalty for going off track
        # How severe should the penalty be? Should it end the episode?
        # off_track_penalty = ???
        
        # TODO: Combine rewards
        # How do you weight different reward components?
        # What happens if rewards are too large or too small?
        # reward = ???
        
        return 0.0  # Placeholder - replace with actual reward calculation
    
    def _is_terminated(self) -> bool:
        """
        Check if the episode should terminate (failure condition).
        
        Questions:
        - What constitutes a failure? (off track, crashed, etc.)
        - Should termination be immediate or gradual?
        
        Returns:
            terminated: True if episode should end due to failure
        """
        # TODO: Check termination conditions
        # When should the episode end in failure?
        # Think about: Distance from track, car orientation, etc.
        # car_pos, car_orn = ???
        # is_off_track = ???
        
        return False  # Placeholder
    
    def _is_truncated(self) -> bool:
        """
        Check if the episode should be truncated (time limit).
        
        Note: Truncation is different from termination!
        - Termination = failure (agent did something wrong)
        - Truncation = time limit (episode just ran out of time)
        
        Returns:
            truncated: True if episode should end due to time limit
        """
        # TODO: Check if max steps reached
        return self.current_step >= self.max_episode_steps
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Questions:
        - Should the track be the same every time or randomized?
        - Should the car start at the same position or vary?
        - What information should be returned in the info dict?
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # TODO: Reset step counter
        # self.current_step = ???
        
        # TODO: Reset car to initial position
        # Where should the car start? (from config)
        # p.resetBasePositionAndOrientation(???, ???, ???, physicsClientId=self.physics_client)
        # p.resetBaseVelocity(???, [0, 0, 0], [0, 0, 0], physicsClientId=self.physics_client)
        
        # TODO: Reset tracking variables
        # self.last_position = ???
        # self.progress = ???
        
        # TODO: Optionally randomize track or starting position
        # How can you add variety to training?
        
        observation = self._get_observation()
        info = {}  # TODO: Add useful info (e.g., track seed, starting position)
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        This is the core function that connects actions to outcomes.
        
        Questions:
        - How do actions map to car controls?
        - How many physics steps should run per environment step?
        - What order should things happen? (action -> physics -> observation -> reward)
        
        Args:
            action: Action from the agent
            
        Returns:
            observation: New observation after step
            reward: Reward for this step
            terminated: Whether episode ended due to failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # TODO: Increment step counter
        # self.current_step += ???
        
        # TODO: Convert action to car controls
        # The TankDriveController expects forward_input and turn_input (both -1 to +1)
        # How do you map your action space to these inputs?
        # forward_input = ???
        # turn_input = ???
        
        # TODO: Apply controls to car
        # The controller.update() method expects keyboard keys, but we're using actions
        # How can you adapt the controller or directly set velocities?
        # Hint: You might need to modify how you use the controller, or set velocities directly
        # p.resetBaseVelocity(???, ???, ???, physicsClientId=self.physics_client)
        
        # TODO: Step physics simulation
        # How many physics steps should run? (config has time_step)
        # p.stepSimulation(physicsClientId=self.physics_client)
        
        # TODO: Get new observation
        # observation = ???
        
        # TODO: Calculate reward
        # reward = ???
        
        # TODO: Check termination and truncation
        # terminated = ???
        # truncated = ???
        
        # TODO: Prepare info dict
        # What information might be useful for debugging or logging?
        # info = {"step": self.current_step, ???}
        
        return (
            np.zeros(10),  # Placeholder observation
            0.0,  # Placeholder reward
            False,  # Placeholder terminated
            False,  # Placeholder truncated
            {},  # Placeholder info
        )
    
    def render(self):
        """
        Render the environment.
        
        Questions:
        - What should be rendered? (car, track, camera view?)
        - How do you render in PyBullet?
        - Should rendering affect performance?
        """
        if self.render_mode == "human":
            # TODO: PyBullet GUI rendering
            # How do you render in PyBullet GUI mode?
            # What camera settings might be useful?
            pass
        elif self.render_mode == "rgb_array":
            # TODO: Return RGB array for video recording
            # How do you capture an image from PyBullet?
            # Hint: p.getCameraImage()
            # return ???
            return None
    
    def close(self):
        """
        Clean up resources.
        
        Questions:
        - What needs to be cleaned up?
        - What happens if you don't close properly?
        """
        # TODO: Disconnect from PyBullet
        # p.disconnect(physicsClientId=self.physics_client)
        pass

