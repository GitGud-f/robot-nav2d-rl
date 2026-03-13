"""
File: nav2d/agents/ppo/buffer.py

Description:
    Implements the RolloutBuffer for the PPO agent.
    Stores on-policy trajectories and calculates Generalized Advantage Estimations (GAE).
"""
import torch
import numpy as np

class RolloutBuffer:
    """Stores on-policy rollouts and computes advantages/returns."""
    def __init__(self, device: torch.device):
        """
        Initializes the RolloutBuffer.
        
        Args: 
            - device (torch.device): The device to store tensors on (CPU or GPU).
        """
        self.device = device
        self.clear()

    def clear(self):
        """
        Clears the buffer after a training update.
        """
        self.states = []
        self.actions =[]
        self.logprobs = []
        self.rewards = []
        self.values =[]
        self.dones =[]

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Calculates Generalized Advantage Estimation (GAE) and Returns.
        
        Args:
            - last_value (float): The value estimation of the state after the final step.
            - gamma (float): Discount factor.
            - gae_lambda (float): GAE smoothing parameter.
        """
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae_lam = 0
        
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
                
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + np.array(self.values)
        
        # Convert to tensors
        self.returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        self.states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        self.actions_tensor = torch.LongTensor(self.actions).to(self.device)
        self.logprobs_tensor = torch.FloatTensor(self.logprobs).to(self.device)
        self.values_tensor = torch.FloatTensor(self.values).to(self.device)
        
        # Advantage Normalization
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)