import os

root = os.path.dirname(os.path.abspath(__file__))

#* Environment parameters

wall_collision_threshold = 0.04
obj_collision_threshold = 0.04
scale = 640
map_size = (scale, scale)

robot_vel_scale = 0.02
creature_vel_scale = 0.01  

lidar_resolution = 200
lidar_max_dist = 0.2 

reach_goal_reward = 100.0
collision_penalty = -50.0
step_penalty = -0.1
max_steps_per_episode = 500


#* DQN Hyperparameterss

dqn_num_episodes = 1000
dqn_lr = 1e-3
dqn_batch_size = 64
dqn_gamma = 0.99
dqn_tau = 0.005
dqn_buffer_size = 100_000
dqn_epsilon_start = 1.0
dqn_epsilon_end = 0.05
dqn_epsilon_decay = 0.995

#* PPO Hyperparameters

ppo_num_episodes = 2000
ppo_rollout_steps = 2048        
ppo_lr_actor = 3e-4              
ppo_lr_critic = 1e-3             
ppo_gamma = 0.99                
ppo_gae_lambda = 0.95            
ppo_clip_epsilon = 0.2           
ppo_epochs = 10                 
ppo_batch_size = 64        
ppo_entropy_coef = 0.01         