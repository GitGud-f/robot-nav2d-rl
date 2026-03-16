"""
PPO Training Script
Trains a PPO agent on the MobileRobotEnv environment.
"""
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nav2d import config
from nav2d.envs.nav_env import MobileRobotEnv
from nav2d.agents.ppo.agent import PPOAgent
from nav2d.utils.logger import MetricLogger
from nav2d.utils.visualizer import plot_learning_curve

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training PPO on device: {device}")

    env = MobileRobotEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device,
        lr_actor=config.ppo_lr_actor,
        lr_critic=config.ppo_lr_critic,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_epsilon=config.ppo_clip_epsilon,
        epochs=config.ppo_epochs,
        batch_size=config.ppo_batch_size,
        entropy_coef=config.ppo_entropy_coef,
        rollout_steps=config.ppo_rollout_steps
    )

    logger = MetricLogger("output/logs/ppo_training")
    
    num_episodes = config.ppo_num_episodes 
    reward_history =[]
    
    global_step = 0
    
    def get_curriculum_level(current_step, max_steps):
        progress = current_step / max_steps
        if progress < 0.20:
            return 1
        elif progress < 0.50:
            return 2
        else:
            return 3
        
    
    for episode in range(1, num_episodes + 1):
        

        current_level = get_curriculum_level(episode, num_episodes)   
        obs, _ = env.reset(curriculum_level=current_level)
        done = False
        episode_reward = 0
        steps = 0
        total_loss = 0
        training_triggered = 0

        while not done and steps < config.max_steps_per_episode:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            loss = agent.step(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            steps += 1
            global_step += 1
            
            if loss is not None:
                total_loss += loss
                training_triggered += 1

        reward_history.append(episode_reward)
        avg_loss = total_loss / training_triggered if training_triggered > 0 else 0
        
        logger.log_episode(episode_reward, steps, avg_loss)
        
        if episode % config.ppo_update_agent == 0:
            agent.train()
            
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode:04d} | Global Steps: {global_step} | Avg Reward (last 10): {avg_reward:7.2f}")
            
            if current_level == 3 and avg_reward > config.early_stopping_avg_reward:
                print(f"\nEnvironment Solved at Episode {episode}!")
                break

        if episode % 100 == 0:
            os.makedirs("output/models", exist_ok=True)
            agent.save(f"output/models/ppo_checkpoint_{episode}.pth")
            plot_learning_curve(reward_history, "output/logs/ppo_learning_curve.png")

    agent.save("output/models/ppo_final.pth")
    logger.close()
    print("PPO Training Completed!")

if __name__ == "__main__":
    main()