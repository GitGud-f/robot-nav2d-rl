"""
File: nav2d/envs/sensors.py

Description: 
    Implements the Lidar sensor simulation for the 2D navigation environment.
    The Lidar casts rays in 360 degrees around the robot to detect obstacles and walls,
    returning normalized distance readings for use in the agent's observation space.
"""

import numpy as np
from nav2d import config

def get_lidar_readings(robot_x: float, robot_y: float, robot_theta: float, obstacles: list) -> np.ndarray:
    """
    Casts 200 rays evenly over 360 degrees.
    Returns an array of shape (200,) with normalized distances [0, 1].
    Distance 1.0 means no obstacle detected within lidar_max_dist.
    
    Args:
        robot_x (float): X coordinate of the robot.
        robot_y (float): Y coordinate of the robot.
        robot_theta (float): Current heading orientation of the robot.
        obstacles (list): List of obstacle objects to check intersections with.
    
    Returns:
        np.ndarray: Normalized Lidar distances across 360 degrees.
    """
    num_rays = config.lidar_resolution
    max_d = config.lidar_max_dist
    radius = config.obj_collision_threshold
    
    angles = robot_theta + np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    ray_dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1) 
    
    distances = np.full(num_rays, max_d)
    origin = np.array([robot_x, robot_y])
    
    for obs in obstacles:
        center = np.array([obs.x, obs.y])
        V = origin - center 
        
        a = 1.0 
        b = 2.0 * np.dot(ray_dirs, V)
        c = np.dot(V, V) - radius**2
        
        discriminant = b**2 - 4 * a * c
        valid_mask = discriminant >= 0
        
        if np.any(valid_mask):
            sqrt_disc = np.sqrt(discriminant[valid_mask])
            t1 = (-b[valid_mask] - sqrt_disc) / 2.0
            
            hit_mask = (t1 > 0) & (t1 < distances[valid_mask])
            distances[valid_mask] = np.where(hit_mask, t1, distances[valid_mask])

    t_x = np.where(ray_dirs[:, 0] > 0, (1.0 - origin[0]) / ray_dirs[:, 0], 
                   np.where(ray_dirs[:, 0] < 0, (0.0 - origin[0]) / ray_dirs[:, 0], np.inf))
    
    t_y = np.where(ray_dirs[:, 1] > 0, (1.0 - origin[1]) / ray_dirs[:, 1], 
                   np.where(ray_dirs[:, 1] < 0, (0.0 - origin[1]) / ray_dirs[:, 1], np.inf))
    
    wall_dist = np.minimum(t_x, t_y)
    distances = np.minimum(distances, wall_dist)
    
    return distances / max_d