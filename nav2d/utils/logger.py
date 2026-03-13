"""
File: nav2d/utils/logger.py

Description: 
    Utility for logging training metrics to TensorBoard.
"""
from torch.utils.tensorboard import SummaryWriter
import os

class MetricLogger:
    """
    Utility class for logging training metrics to TensorBoard.
    """
    def __init__(self, log_dir: str):
        """
        Initializes a TensorBoard writer.
        Args:
            - log_dir (str): Directory where TensorBoard logs will be saved.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.episode = 0

    def log_episode(self, reward: float, steps: int, loss: float = None, epsilon: float = None):
        """
        Logs metrics for a training episode.
        
        Args:
            - reward (float): Total reward obtained in the episode.
            - steps (int): Number of steps taken in the episode.
            - loss (float, optional): Loss value from training. Defaults to None.
            - epsilon (float, optional): Current epsilon value for exploration. Defaults to None.
        """
        self.episode += 1
        self.writer.add_scalar('Train/Reward', reward, self.episode)
        self.writer.add_scalar('Train/Steps', steps, self.episode)
        
        if loss is not None:
            self.writer.add_scalar('Train/Loss', loss, self.episode)
        if epsilon is not None:
            self.writer.add_scalar('Train/Epsilon', epsilon, self.episode)

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()