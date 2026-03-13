"""
File: nav2d/envs/elements.py

Description: 
    Defines the core physical entities in the 2D navigation environment, 
    including the robot, goal, static obstacles, and moving creatures. 
    Each class manages its own position, orientation, and rendering information.
    
Classes:
    - ObjectBase: Base class for all physical entities, handling common attributes and rendering.
    - VelRobot: Represents the mobile robot, allowing movement based on velocity commands.
    - Goal: Represents the target location the robot must reach.
    - StaticObstacle: Represents immobile obstacles that the robot must avoid.
    - RandomPathCreature: Represents a moving creature that follows a predefined path.
    - OrbitingCreature: Represents a creature that orbits around the goal.
"""

import numpy as np
import pygame
from nav2d import config

class ObjectBase:
    """
    Base class representing any physical entity in the simulation.
    Manages position, orientation, and rendering information.
    """
    def __init__(self, image_path: str, shape: np.ndarray, init_x: float, init_y: float):
        """
        Initializes the base object with an image and initial coordinates.
        
        Args:
            - image_path (str): Path to the image representing the object.
            - shape (np.ndarray): Tuple representing the width and height of the object in pixels.
            - init_x (float): Initial X coordinate (normalized [0, 1]).
            - init_y (float): Initial Y coordinate (normalized [0, 1]).
        """
        
        self.image = pygame.image.load(image_path)
        self.shape = shape
        self.image = pygame.transform.scale(self.image, tuple(self.shape))
        self.x, self.y = init_x, init_y
        self.orient = np.pi / 2

    def render_info(self, scale: int) -> tuple:
        """
        Returns the image and position for rendering the object.
        
        Args:
            - scale (int): Scaling factor for the rendering.
            
        Returns:
            - tuple: The image and its position for rendering.
        """
        return self.image, (int(scale * self.x - self.shape[0] / 2), 
                            config.map_size[1] - int(scale * self.y - self.shape[1] / 2))

class VelRobot(ObjectBase):
    """
    Represents the mobile robot in the environment, allowing movement based on velocity commands.
    """
    
    def __init__(self, init_x: float, init_y: float):
        """
        Initializes the robot with a specific image and starting coordinates.
        
        Args:
            - init_x (float): Initial X coordinate (normalized [0, 1]).
            - init_y (float): Initial Y coordinate (normalized [0, 1]).
        """
        
        super().__init__(f"{config.root}/../assets/robot.png", np.array((30, 30)), init_x, init_y)

    def move(self, dx: float, dy: float):
        """
        Moves the robot by the specified deltas, clamped to the map boundaries.
        
        Args:
            - dx (float): Delta X coordinate.
            - dy (float): Delta Y coordinate.
        """ 
        self.x = np.clip(self.x + dx, 0, 1)
        self.y = np.clip(self.y + dy, 0, 1)

class Goal(ObjectBase):
    """
    Represents the target location the robot must reach.
    """
    def __init__(self, x: float, y: float):
        """
        Initializes the goal with a specific image and coordinates. 
        
        Args:            
            - x (float): X coordinate of the goal (normalized [0, 1]).
            - y (float): Y coordinate of the goal (normalized [0, 1]).
        """
        super().__init__(f"{config.root}/../assets/goal.png", np.array((30, 30)), x, y)

class StaticObstacle(ObjectBase):
    """
    Represents a static obstacle in the environment.
    """
    def __init__(self, x: float, y: float):
        """
        Initializes the static obstacle with a specific image and coordinates.
        
        Args:
            - x (float): X coordinate of the obstacle (normalized [0, 1]).
            - y (float): Y coordinate of the obstacle (normalized [0, 1]).
        """
        
        super().__init__(f"{config.root}/../assets/cat.png", np.array((30, 30)), x, y)

class RandomPathCreature(ObjectBase):
    """
    Represents a creature that follows a random path.
    """
    def __init__(self, waypoints, velocity=config.creature_vel_scale):
        """ 
        Initializes the random path creature with a specific image, waypoints, and velocity.
        
        Args:
            - waypoints (list): List of (x, y) tuples representing the path waypoints.
            - velocity (float): Speed at which the creature moves along the path.
        """
        super().__init__(f"{config.root}/../assets/cat.png", np.array((30, 30)), waypoints[0][0], waypoints[0][1])
        self.waypoints = waypoints
        self.vel = velocity
        self.target_idx = 1

    def step(self):
        """
        Moves the creature towards the current target waypoint, updating its position.
        If the creature reaches the waypoint, it updates to the next target in the list.
        """
        
        target = self.waypoints[self.target_idx]
        dx, dy = target[0] - self.x, target[1] - self.y
        dist = np.hypot(dx, dy)
        
        if dist < self.vel: # Reached waypoint
            self.target_idx = (self.target_idx + 1) % len(self.waypoints)
        else:
            self.x += (dx / dist) * self.vel
            self.y += (dy / dist) * self.vel

class OrbitingCreature(ObjectBase):
    """
    Represents a creature that orbits around a central goal.
    """
    def __init__(self, goal: Goal, orbit_radius: float = 0.1, velocity: float = config.creature_vel_scale):
        """ 
        Initializes the orbiting creature with a specific image, goal, and orbit parameters.
        
        Args:
            - goal (Goal): The central goal around which the creature will orbit.
            - orbit_radius (float): The radius of the orbit.
            - velocity (float): The speed at which the creature moves along the orbit.
        """
        init_x = goal.x + orbit_radius
        init_y = goal.y
        super().__init__(f"{config.root}/../assets/cat.png", np.array((30, 30)), init_x, init_y)
        self.goal = goal
        self.radius = orbit_radius
        self.angle = 0.0

        self.angular_vel = velocity / orbit_radius 

    def step(self):
        """ 
        Updates the creature's position by moving it along its circular orbit around the goal.
        """
        self.angle = (self.angle + self.angular_vel) % (2 * np.pi)
        self.x = self.goal.x + self.radius * np.cos(self.angle)
        self.y = self.goal.y + self.radius * np.sin(self.angle)