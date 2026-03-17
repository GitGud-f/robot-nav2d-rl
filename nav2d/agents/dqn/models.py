"""
File: nav2d/agents/dqn/models.py

Description:
    Defines the neural network architecture for the Deep Q-Network (DQN) 
    agent used in the 2D navigation environment. 
    The QNetwork class implements a simple feedforward neural network 
    that takes the state as input and outputs Q-values for each possible action.
"""
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Neural network architecture for the DQN agent."""
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initializes the QNetwork with the specified state and action dimensions.
        
        Args:
            - state_dim (int): Dimension of the state space.
            - action_dim (int): Dimension of the action space.
        """
        
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the QNetwork.
        
        Args:
            - x (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
            
        Returns:
            - torch.Tensor: Output Q-values for each action, shape (batch_size, action_dim)
        """
        return self.net(x)