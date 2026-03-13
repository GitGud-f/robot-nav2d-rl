import os

root = os.path.dirname(os.path.abspath(__file__))

wall_collision_threshold = 0.04
obj_collision_threshold = 0.04
scale = 600
map_size = (scale, scale)


robot_vel_scale = 0.02
creature_vel_scale = 0.01  


lidar_resolution = 200
lidar_max_dist = 0.2 


reach_goal_reward = 100.0
collision_penalty = -50.0
step_penalty = -0.1