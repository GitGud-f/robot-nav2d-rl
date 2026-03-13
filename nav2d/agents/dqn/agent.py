import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from nav2d.agents.base_agent import BaseAgent
from nav2d.agents.dqn.models import QNetwork
from nav2d.agents.dqn.buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.005, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 buffer_size=100000, batch_size=64, device="cpu"):
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

    def select_action(self, state, evaluate=False):
        # Epsilon-greedy action selection
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        """Saves experience and triggers training if buffer is large enough."""
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            loss = self.train()
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return loss
        return None

    def train(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Compute current Q values
        current_q = self.q_net(states).gather(1, actions)

        # Compute Target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute Loss (Huber Loss is more stable than MSE for RL)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network: θ' = τθ + (1 - τ)θ'
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def save(self, filepath):
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath):
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())