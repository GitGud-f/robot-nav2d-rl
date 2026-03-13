"""
File: nav2d/agents/ppo/models.py

Description:
    Defines the neural network architectures for the PPO agent.
    Contains an Actor network for policy approximation and a Critic network for state-value estimation.
    Uses Orthogonal Initialization, which is highly recommended for PPO performance.
"""

import torch
import torch.nn as nn
import numpy as np

def layer_init(layer: nn.Linear, std: float=np.sqrt(2), bias_const: float=0.0) -> nn.Linear:
    """
    Applies orthogonal initialization to the layers.

    Args:
        - layer (nn.Linear): The layer to initialize.
        - std (float): Standard deviation for the orthogonal initialization.
        - bias_const (float): Constant value to initialize the biases.
        
    Returns:
        - nn.Linear: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    """Actor network for predicting action distributions."""
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initializes the ActorNetwork.
        
        Args:
            - state_dim (int): Dimension of the input state space.
            - action_dim (int): Number of discrete actions in the action space.
        """
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.

        Args: 
            - x (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
            
        Returns:
            - torch.Tensor: Logits for action probabilities of shape (batch_size, action_dim).
        """
        return self.network(x)

class CriticNetwork(nn.Module):
    """
    Critic network for estimating state values.
    """
    def __init__(self, state_dim: int):
        """
        Initializes the CriticNetwork.
        
        Args:
            - state_dim (int): Dimension of the input state space.
        """
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            - x (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            - torch.Tensor: Estimated state values of shape (batch_size, 1).
        """
        return self.network(x)