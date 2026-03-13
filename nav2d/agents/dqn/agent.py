"""
File: nav2d/agents/dqn/agent.py

Description:
    Implements the DQNAgent class, which is a Deep Q-Network (DQN) agent for the 2D navigation environment.
    The DQNAgent uses a neural network to approximate the Q-function, a replay buffer to store experience tuples, 
    and an epsilon-greedy strategy for action selection. 
    The agent can be trained using sampled experiences from the replay buffer and can save/load its model weights for persistence.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from nav2d.agents.base_agent import BaseAgent
from nav2d.agents.dqn.models import QNetwork
from nav2d.agents.dqn.buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent for the 2D navigation environment.
    The DQNAgent uses a neural network to approximate the Q-function, a replay buffer to store experience tuples, 
    and an epsilon-greedy strategy for action selection.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float=1e-3, gamma: float=0.99, tau: float=0.005, 
                 epsilon_start: float=1.0, epsilon_end: float=0.01, epsilon_decay: float=0.995, 
                 buffer_size: int=100000, batch_size: int=64, device: str="cpu"):
        """
        Initializes the DQNAgent with the specified parameters.
        
        Args:
            - state_dim (int): Dimension of the state space.
            - action_dim (int): Dimension of the action space.
            - lr (float): Learning rate for the optimizer.
            - gamma (float): Discount factor for future rewards.
            - tau (float): Soft update parameter for target network.
            - epsilon_start (float): Initial value of epsilon for epsilon-greedy action selection.
            - epsilon_end (float): Minimum value of epsilon after decay.
            - epsilon_decay (float): Decay rate for epsilon after each episode.
            - buffer_size (int): Maximum number of experience tuples to store in the replay buffer.
            - batch_size (int): Number of experience tuples to sample for each training step.
            - device (str): Computation device ('cpu' or 'cuda').
        """
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Epsilon-greedy parameters for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, self.device)

    def select_action(self, state: np.ndarray, evaluate=False) -> int:
        """
        Selects an action given the current state using an epsilon-greedy strategy.
        
        Args:
            - state (np.ndarray): The current state.
            - evaluate (bool): Whether to evaluate the action (e.g., for inference).
            
        Returns:
            - int: The selected action.
        """
        
        # Epsilon-greedy action selection
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """
        Saves experience and triggers training if buffer is large enough.
        
        Args:
            - state (np.ndarray): The current state.
            - action (int): The action taken.
            - reward (float): The reward received.
            - next_state (np.ndarray): The next state.
            - done (bool): Whether the episode is done.
            
        Returns:
            - float: The loss from training.
        """
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            loss = self.train()
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return loss
        return None

    def train(self) -> float:
        """
        Trains the DQN agent's model based on sampled experience from the replay buffer.
        
        Returns:
            - float: The loss from the training step.
        """
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def save(self, filepath: str):
        """
        Saves the model weights to a file.
        
        Args:
            - filepath (str): The path to the file where the weights will be saved.
        """
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        """
        Loads the model weights from a file.
        
        Args:
            - filepath (str): The path to the file from which the weights will be loaded.
        """
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())