"""
File: nav2d/utils/visualizer.py

Description:
    Utility functions for visualizing training and evaluation results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import imageio
import numpy as np
import os
from tqdm import tqdm

def plot_learning_curve(rewards: list, filename: str, window: int = 100):
    """
    Plots the learning curve of rewards over episodes and saves it to a file.
    
    Args:
        - rewards: List of total rewards obtained in each episode.
        - filename: Path to save the plot image (e.g., 'outputs/plots/learning_curve.png').
        - window: Window size for calculating the moving average (default: 100).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(10, 6), facecolor="white")
    
    df = pd.DataFrame({'reward': rewards})
    rolling_mean = df['reward'].rolling(window=window, min_periods=1).mean()
    

    plt.plot(rewards, alpha=0.4, color='cyan', label='Episode Reward')
    plt.plot(rolling_mean, color='magenta', linewidth=2, label=f'Moving Average ({window} ep)')
    
    plt.title('Training Learning Curve', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    
    ax = plt.gca()
    ax.set_facecolor("black")
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved to {filename}")


def create_eval_video(env, agent, filename: str, n_trials: int = 3, fps: int = 30):
    """
    Evaluates the agent for n_trials and records the frames to an mp4 file.
    Args:
        - env: The environment to evaluate on.
        - agent: The trained agent to evaluate.
        - filename: Path to save the video (e.g., 'outputs/videos/eval_video.mp4').
        - n_trials: Number of evaluation episodes to record (default: 3).
        - fps: Frames per second for the output video (default: 30).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Recording evaluation video to {filename} ...")
    
    frames =[]
    
    for trial in tqdm(range(n_trials), desc="Recording Episodes"):
        obs, _ = env.reset()
        done = False
        
        # Render the initial frame
        frames.append(env.render())
        
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            frames.append(frame)
            
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Test Video saved successfully at: {filename}")