"""
File: nav2d/agents/ppo/agent.py

Description:
    Implements the PPO Agent, featuring Actor-Critic networks, GAE calculation, 
    and a clipped surrogate objective function.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from nav2d.agents.base_agent import BaseAgent
from nav2d.agents.ppo.models import ActorNetwork, CriticNetwork
from nav2d.agents.ppo.buffer import RolloutBuffer

class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent."""
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float=3e-4, lr_critic: float=1e-3, 
                 gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, 
                 epochs: int=10, batch_size: int=64, entropy_coef: float=0.01,
                 rollout_steps: int=2048, device: str="cpu"):
        """
        Initializes the PPOAgent with Actor-Critic networks and training hyperparameters.
        
        Args: 
            - state_dim (int): Dimension of the input state space.
            - action_dim (int): Number of discrete actions in the action space.
            - lr_actor (float): Learning rate for the actor network.
            - lr_critic (float): Learning rate for the critic network.
            - gamma (float): Discount factor for future rewards.
            - gae_lambda (float): GAE smoothing parameter.
            - clip_epsilon (float): Clipping parameter for PPO surrogate objective.
            - epochs (int): Number of training epochs per update.
            - batch_size (int): Mini-batch size for training updates.
            - entropy_coef (float): Coefficient for entropy regularization to encourage exploration.
            - rollout_steps (int): Number of steps to collect before each training update.
            - device (str): Device to run the computations on ("cpu" or "cuda").
        """
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.memory = RolloutBuffer(self.device)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state: np.ndarray, evaluate: bool=False) -> int:
        """
        Selects an action using the current policy and stores transition data for training.
        
        Args:
            - state (np.ndarray): The current state of the environment.
            - evaluate (bool): If True, selects the action with the highest probability (greedy).
        
        Returns: 
            - int: The selected action index.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
        dist = Categorical(logits=logits)
        
        if evaluate:
            # Greedily pick the action with highest probability during evaluation
            return torch.argmax(logits, dim=1).item()
            
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        self.memory.states.append(state)
        self.memory.actions.append(action.item())
        self.memory.logprobs.append(action_logprob.item())
        self.memory.values.append(value.item())
        
        return action.item()

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """
        Saves step environment rewards and handles rollout completion.
        
        Args:
            - state (np.ndarray): The current state of the environment.
            - action (int): The action taken by the agent.
            - reward (float): The reward received after taking the action.
            - next_state (np.ndarray): The next state of the environment after taking the action.
            - done (bool): Whether the episode has ended after this step.
            
        Returns:
            - float: The loss from training (if training is triggered), otherwise None.
        """
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)
        
        # Once buffer is full, trigger training
        if len(self.memory.states) >= self.rollout_steps:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                last_value = self.critic(next_state_tensor).item()
                
            loss = self.train(last_value)
            return loss
            
        return None

    def train(self, last_value: float) -> float:
        """
        Trains the agent over the collected rollout buffer.
        
        Args: 
            - last_value (float): The value estimation of the state after the final step of the rollout.
        
        Returns: 
            - float: The average loss over the training epochs.
        """
        self.memory.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        b_states = self.memory.states_tensor
        b_actions = self.memory.actions_tensor
        b_logprobs = self.memory.logprobs_tensor
        b_advantages = self.memory.advantages
        b_returns = self.memory.returns
        
        dataset_size = len(b_states)
        indices = np.arange(dataset_size)
        
        total_loss = 0.0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]
                
                # Forward pass for mini-batch
                logits = self.actor(b_states[mb_idx])
                values = self.critic(b_states[mb_idx]).squeeze()
                
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()
                
                # Ratio for PPO Surrogate
                logratio = new_logprobs - b_logprobs[mb_idx]
                ratio = logratio.exp()
                
                # Clipped Surrogate Objective
                mb_advantages = b_advantages[mb_idx]
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                critic_loss = self.mse_loss(values, b_returns[mb_idx])
                
                # Total Loss
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                
        self.memory.clear()
        return total_loss / (self.epochs * (dataset_size // self.batch_size))

    def save(self, filepath: str):
        """
        Saves the agent's state to a file.
        
        Args:
            - filepath (str): The path to the file where the state will be saved.
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """
        Loads the agent's state from a file.
        
        Args:
            - filepath (str): The path to the file from which the state will be loaded.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])