import os

root = os.path.dirname(os.path.abspath(__file__))

#* Environment parameters

wall_collision_threshold = 0.04
obj_collision_threshold = 0.04
scale = 640
map_size = (scale, scale)

robot_vel_scale = 0.03
creature_vel_scale = 0.01  

lidar_resolution = 200
lidar_max_dist = 0.2 
min_safe_dist = 0.25

reach_goal_reward = 700.0
too_close_penalty = -0.5
collision_penalty = -200
step_penalty = -0.8
dense_distance_reward = 90
dense_angle_reward = 40
max_steps_per_episode = 300 

early_stopping_avg_reward = 95.0

#* DQN Hyperparameterss

dqn_num_episodes = 5000
dqn_lr = 1e-4
dqn_batch_size = 64
dqn_gamma = 0.99
dqn_tau = 0.005
dqn_buffer_size = 200_000
dqn_epsilon_start = 1.0
dqn_epsilon_end = 0.05
dqn_epsilon_decay = 0.995

#* PPO Hyperparameters

ppo_num_episodes = 5000
ppo_update_agent = 50
ppo_rollout_steps = 2048        
ppo_lr_actor = 3e-4              
ppo_lr_critic = 1e-3             
ppo_gamma = 0.99                
ppo_gae_lambda = 0.95            
ppo_clip_epsilon = 0.2           
ppo_epochs = 10                 
ppo_batch_size = 64        
ppo_entropy_coef = 0.01        