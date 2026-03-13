"""
File: nav2d/envs/nav_env.py 

Description:
    Defines the MobileRobotEnv class, a custom OpenAI Gym environment for a 2D mobile robot navigation task.
    The environment includes a robot, a goal, static obstacles, and moving creatures.
    The robot receives observations from a Lidar sensor and must learn to navigate to the goal while avoiding collisions.
    The environment supports four discrete actions: turn right, turn left, move forward, and sprint forward.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from nav2d.envs.elements import VelRobot, Goal, StaticObstacle, RandomPathCreature, OrbitingCreature
from nav2d.envs.engine import NavigationEngine
from nav2d.envs.sensors import get_lidar_readings
from nav2d import config

class MobileRobotEnv(gym.Env):
    """
    A reinforcement learning environment for a 2D mobile robot navigation task.
    The robot must navigate to a goal while avoiding static and moving obstacles.
    """
    def __init__(self):
        """
        Initializes the environment, defining action and observation spaces.
        """
        super(MobileRobotEnv, self).__init__()
        
        # Actions: Right, Left, Forward, Sprint
        self.action_space = spaces.Discrete(4)
        
        # Obs: [robot_x, robot_y, goal_x, goal_y] + 200 Lidar rays
        obs_dim = 4 + config.lidar_resolution
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.engine = None
        self.max_steps = config.max_steps_per_episode
        self.current_step = 0

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        Resets the environment to an initial state, randomizing the positions of the robot, goal, and obstacles.
        
        Args:
            - seed (int, optional): Random seed for reproducibility.
            -  options (dict, optional): Additional options for environment configuration.
            
        Returns:
            - tuple: (observation, info)
        """
        
        super().reset(seed=seed)
        self.current_step = 0
        
        robot = VelRobot(np.random.uniform(0.1, 0.3), np.random.uniform(0.1, 0.3))
        goal = Goal(np.random.uniform(0.7, 0.9), np.random.uniform(0.7, 0.9))
        
        static_obs =[
            StaticObstacle(0.5, 0.5),
            StaticObstacle(0.3, 0.8),
            StaticObstacle(0.8, 0.3)
        ]
        
        moving_creatures =[
            OrbitingCreature(goal, orbit_radius=0.1),
            RandomPathCreature(waypoints=[(0.2, 0.5), (0.8, 0.5), (0.5, 0.9)])
        ]
        
        self.engine = NavigationEngine(robot, goal, static_obs, moving_creatures)
        
        return self._get_obs(), {}

    def step(self, action: int) -> tuple:
        """
        Advances the environment by one step given the agent's action.
        
        Args:
            - action (int): The action to take (0: Right, 1: Left, 2: Forward, 3: Sprint).
            
        Returns:
            - tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        prev_dist = np.hypot(self.engine.robot.x - self.engine.goal.x, 
                             self.engine.robot.y - self.engine.goal.y)
        
        self.engine.step_physics(action)
        
        hit_obstacle, reach_goal = self.engine.check_collisions()
        
        new_dist = np.hypot(self.engine.robot.x - self.engine.goal.x, 
                            self.engine.robot.y - self.engine.goal.y)
                            
        reward = config.step_penalty
        reward += (prev_dist - new_dist) * 100 # Dense reward for moving closer
        
        terminated = False
        if reach_goal:
            reward += config.reach_goal_reward
            terminated = True
        elif hit_obstacle:
            reward += config.collision_penalty
            terminated = True
            
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation vector for the current state of the environment.
        The observation includes the robot's position, the goal's position, and Lidar sensor readings
        for obstacle detection.
        
        Returns: 
            - np.ndarray: The observation vector.
        """
        
        base_state = np.array([
            self.engine.robot.x, self.engine.robot.y,
            self.engine.goal.x, self.engine.goal.y
        ], dtype=np.float32)
        
        lidar_readings = get_lidar_readings(
            self.engine.robot.x, self.engine.robot.y, 
            self.engine.robot.orient, self.engine.all_obstacles
        )
        
        return np.concatenate((base_state, lidar_readings))

    def render(self) -> np.ndarray:
        """
        Renders the current state of the environment using Pygame.
        
        Returns:
            - np.ndarray: The rendered image as a NumPy array.
        """
        return self.engine.render()