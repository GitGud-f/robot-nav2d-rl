"""
DQN Training Script
This script trains a DQN agent on the MobileRobotEnv environment. 
It logs training metrics to TensorBoard and saves model checkpoints every 100 episodes. 
The learning curve is also plotted and saved as an image.
"""
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nav2d import config
from nav2d.envs.nav_env import MobileRobotEnv
from nav2d.agents.dqn.agent import DQNAgent
from nav2d.utils.logger import MetricLogger
from nav2d.utils.visualizer import plot_learning_curve

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training DQN on device: {device}")

    env = MobileRobotEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device,
        lr=config.dqn_lr,
        gamma=config.dqn_gamma,
        tau=config.dqn_tau,
        batch_size=config.dqn_batch_size,
        buffer_size=config.dqn_buffer_size,
        epsilon_start=config.dqn_epsilon_start,
        epsilon_end=config.dqn_epsilon_end,
        epsilon_decay=config.dqn_epsilon_decay
    )

    logger = MetricLogger("output/logs/dqn_training")
    
    num_episodes = 1000
    reward_history =[]

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        total_loss = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            loss = agent.step(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            steps += 1
            if loss is not None:
                total_loss += loss

        reward_history.append(episode_reward)
        avg_loss = total_loss / steps if steps > 0 else 0
        
        logger.log_episode(episode_reward, steps, avg_loss, agent.epsilon)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode:04d} | Avg Reward (last 10): {avg_reward:7.2f} | Epsilon: {agent.epsilon:.3f}")

        if episode % 100 == 0:
            os.makedirs("output/models", exist_ok=True)
            agent.save(f"output/models/dqn_checkpoint_{episode}.pth")
            plot_learning_curve(reward_history, "output/logs/dqn_learning_curve.png")

    agent.save("output/models/dqn_final.pth")
    logger.close()
    print("DQN Training Completed!")

if __name__ == "__main__":
    main()