"""
File: nav2d/agents/dqn/buffer.py

Description:
    Implements the ReplayBuffer class for storing and sampling experience tuples in the DQN agent.
    The ReplayBuffer uses a deque to store a fixed number of experience tuples and provides a method
    for sampling random batches of experiences for training the DQN.
"""
import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for DQN agent."""
    def __init__(self, capacity: int, device: torch.device):
        """
        Initializes the replay buffer with a specified capacity and computation device.
        
        Args:
            - capacity (int): Maximum number of experience tuples to store in the buffer.
            - device (torch.device): Computation device for storing sampled batches (e.g., 'cpu' or 'cuda').
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """
        Adds an experience tuple to the replay buffer.
        
        Args:
            - state (torch.Tensor): The current state.
            - action (int): The action taken.
            - reward (float): The reward received after taking the action.
            - next_state (torch.Tensor): The next state after taking the action.
            - done (bool): Whether the episode has ended after taking the action.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Samples a random batch of experiences from the replay buffer.
        
        Args:
            - batch_size (int): The number of experiences to sample.
            
        Returns:
            - tuple: A tuple containing the sampled experiences (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).unsqueeze(1).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        """
        Returns the current number of experience tuples stored in the buffer.
        """
        return len(self.buffer)