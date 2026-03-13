"""
File: nav2d/envs/engine.py

Description: 
    Core physics and rendering engine for the 2D navigation environment.
    Handles movement, collision detection, and rendering using Pygame.
"""
import pygame
import numpy as np

from nav2d import config
from nav2d.envs.elements import Goal, VelRobot

class NavigationEngine:
    """
    Core physics and rendering engine for the 2D navigation environment.
    Handles movement, collision detection, and rendering using Pygame.
    """
    def __init__(self, robot: VelRobot, goal: Goal, static_obs: list, moving_creatures: list):
        """
        Initializes the simulation engine with the specified entities.
        
        Args:
            - robot (VelRobot): The mobile robot instance.
            - goal (Goal): The target location instance.
            - static_obs (list): List of StaticObstacle instances.
            - moving_creatures (list): List of moving creature instances.
        """
        self.screen = None
        pygame.display.set_caption("2D Mobile Robot RL Navigation")
        
        self.robot = robot
        self.goal = goal
        self.static_obs = static_obs
        self.moving_creatures = moving_creatures
        
        self.all_obstacles = self.static_obs + self.moving_creatures

    def step_physics(self, action: int):
        """
        Updates the physics of the simulation based on the given action.
        
        Args:
            - action (int): The action to take (0: Right, 1: Left, 2: Forward, 3: Sprint).
        """
        if action == 0: # RIGHT
            self.robot.orient -= np.pi/4
        elif action == 1: # LEFT
            self.robot.orient += np.pi/4
        elif action == 2: # FORWARD
            self.robot.move(config.robot_vel_scale * np.cos(self.robot.orient),
                            config.robot_vel_scale * np.sin(self.robot.orient))
        elif action == 3: # SPRINT
            self.robot.move(2 * config.robot_vel_scale * np.cos(self.robot.orient),
                            2 * config.robot_vel_scale * np.sin(self.robot.orient))

        self.robot.orient = self.robot.orient % (2 * np.pi)

        for creature in self.moving_creatures:
            creature.step()

    def check_collisions(self) -> tuple:
        """
        Checks for collisions with walls, obstacles, and goal.
        
        Returns:
            - tuple: (hit_obstacle (bool), reach_goal (bool))
        """
        # Wall Collision
        if self.robot.x <= 0 or self.robot.x >= 1 or self.robot.y <= 0 or self.robot.y >= 1:
            return True, False
        
        # Goal Reached
        if np.hypot(self.robot.x - self.goal.x, self.robot.y - self.goal.y) <= config.obj_collision_threshold:
            return False, True
            
        # Obstacle Collision
        for obs in self.all_obstacles:
            if np.hypot(self.robot.x - obs.x, self.robot.y - obs.y) <= config.obj_collision_threshold:
                return True, False
                
        return False, False

    def render(self) -> np.ndarray:
        """
        Renders the current state of the environment using Pygame.
        
        Returns: 
            - np.ndarray: The rendered image as a NumPy array.
        """
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(config.map_size)
            
        self.screen.fill((155, 255, 255))
        
        for obs in self.all_obstacles:
            self.screen.blit(*obs.render_info(scale=config.scale))
            
        self.screen.blit(*self.goal.render_info(scale=config.scale))
        
        robot_image, robot_pos = self.robot.render_info(scale=config.scale)
        robot_image = pygame.transform.rotate(robot_image, self.robot.orient * 180 / np.pi - 90)
        self.screen.blit(robot_image, robot_pos)
        
        pygame.display.update()
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))