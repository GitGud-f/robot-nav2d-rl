"""
File: scripts/evaluate.py

Description:
    Evaluates a trained DQN or PPO agent on the MobileRobotEnv.
    Calculates performance metrics (success rate, collision rate, average reward).
    Optionally streams ALL evaluated episodes directly to an MP4 video file 
    without overloading system RAM.
"""

import os
import sys
import argparse
import torch
import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nav2d import config
from nav2d.envs.nav_env import MobileRobotEnv
from nav2d.agents.dqn.agent import DQNAgent
from nav2d.agents.ppo.agent import PPOAgent

def evaluate_agent(env: MobileRobotEnv, agent: DQNAgent or PPOAgent, num_episodes: int, record: bool = False, video_path: str = None):
    """
    Runs the agent through the environment for a given number of episodes.
    Calculates statistical metrics and optionally records all runs to a video.
    
    Args:
        - env (MobileRobotEnv): The environment to evaluate on.
        - agent (DQNAgent or PPOAgent): The trained agent to evaluate.
        - num_episodes (int): The number of episodes to run for evaluation.
        - record (bool): Whether to record a video of the evaluation runs.
        - video_path (str): The file path to save the recorded video (if record=True).
    """
    print(f"Starting evaluation over {num_episodes} episodes...")
    
    writer = None
    if record:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        print(f"Streaming video directly to {video_path} (Optimized for low RAM usage)...")
        writer = imageio.get_writer(video_path, fps=30)
    
    successes = 0
    collisions = 0
    timeouts = 0
    total_rewards =[]
    steps_taken =[]

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        if record:
            writer.append_data(env.render())
            
        while not done:
            # agent.epsilon  = 0.20
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if record:
                writer.append_data(env.render())
                
            episode_reward += reward
            steps += 1
            
            if done:
                dist_to_goal = np.hypot(env.engine.robot.x - env.engine.goal.x, 
                                        env.engine.robot.y - env.engine.goal.y)
                
                if dist_to_goal <= config.obj_collision_threshold:
                    successes += 1
                elif truncated:
                    timeouts += 1
                else:
                    collisions += 1

        total_rewards.append(episode_reward)
        steps_taken.append(steps)
        
        if ep % 10 == 0:
            print(f"Episode {ep:03d}/{num_episodes} | Reward: {episode_reward:7.2f} | Steps: {steps}")

    if record:
        writer.close()
        print(f"Video of all {num_episodes} episodes successfully saved to: {video_path}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(steps_taken)
    success_rate = (successes / num_episodes) * 100
    collision_rate = (collisions / num_episodes) * 100
    timeout_rate = (timeouts / num_episodes) * 100

    print("\n" + "="*50)
    print("               EVALUATION RESULTS               ")
    print("="*50)
    print(f"Total Episodes : {num_episodes}")
    print(f"Success Rate   : {success_rate:.2f}% ({successes}/{num_episodes})")
    print(f"Collision Rate : {collision_rate:.2f}% ({collisions}/{num_episodes})")
    print(f"Timeout Rate   : {timeout_rate:.2f}% ({timeouts}/{num_episodes})")
    print(f"Average Reward : {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps  : {avg_steps:.1f}")
    print("="*50 + "\n")

def main():
    """
    Main function to parse arguments, load the trained model, and run evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained RL Agent.")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], required=True, 
                        help="Algorithm to evaluate ('dqn' or 'ppo').")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model weights (.pth file).")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes to evaluate (default: 100).")
    parser.add_argument("--record", action="store_true", 
                        help="Flag to record a video of ALL evaluated episodes.")
    parser.add_argument("--video_path", type=str, default="output/videos/eval_video.mp4", 
                        help="Output path for the recorded video.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = MobileRobotEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if args.algo == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    elif args.algo == "ppo":
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")
        
    print(f"Loading {args.algo.upper()} model from {args.model_path}...")
    agent.load(args.model_path)
    
    evaluate_agent(env, agent, args.episodes, record=args.record, video_path=args.video_path)


if __name__ == "__main__":
    main()