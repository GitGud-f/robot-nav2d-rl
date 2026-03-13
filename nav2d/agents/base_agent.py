"""
File: nav2d/agents/base_agent.py

Description:
    Defines the abstract base class for reinforcement learning agents in the 2D navigation environment. 
"""

from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents in the 2D navigation environment.
    Defines the interface for action selection, training, and model persistence.
    """
    def __init__(self, state_dim: int, action_dim: int, device: str):
        """
        Initializes the agent with the specified state and action dimensions, and computation device.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

    @abstractmethod
    def select_action(self, state, evaluate=False):
        """
        Selects an action given the current state.
        Args:
            state: The current state.
            evaluate (bool): Whether to evaluate the action (e.g., for inference).
        Returns:
            The selected action.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Trains the agent's model based on collected experience.
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        Saves the model weights to a file.
        Args:
            filepath (str): The path to the file where the weights will be saved.
        """
        pass

    @abstractmethod
    def load(self, filepath: str):
        """
        Loads the model weights from a file.
        Args:
            filepath (str): The path to the file from which the weights will be loaded.
        """
        pass