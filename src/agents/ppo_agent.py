"""
Proximal Policy Optimization (PPO) Agent for Bipedal Locomotion

This module implements a PPO agent with support for both discrete and continuous
action spaces, designed for single-agent bipedal locomotion tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from collections import deque

from .base_agent import BaseAgent, AgentConfig
from models.neural_networks.policy_networks import PolicyNetwork, ContinuousPolicyNetwork
from models.neural_networks.value_networks import ValueNetwork

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent for bipedal locomotion.
    
    This agent implements the PPO algorithm with support for both discrete
    and continuous action spaces, along with various improvements for
    bipedal locomotion tasks.
    """
    
    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        device: str = "cpu",
        # PPO-specific parameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        # Network architecture
        hidden_dim: int = 128,
        num_layers: int = 2,
        # Action bounds for continuous actions
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        **kwargs
    ):
        """
        Initialize PPO agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            action_type: Type of action space ("discrete" or "continuous")
            device: Device to run computations on
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy loss
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs
            batch_size: Batch size for training
            hidden_dim: Hidden dimension for neural networks
            num_layers: Number of layers in neural networks
            action_bounds: Action bounds for continuous actions
        """
        super().__init__(agent_id, obs_dim, action_dim, action_type, device, **kwargs)
        
        # PPO parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.action_bounds = action_bounds
        
        # Initialize networks
        self._build_networks(hidden_dim, num_layers)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Training statistics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.kl_divergence_history = []
        
        logger.info(f"Initialized PPO agent {agent_id} with {action_type} actions")
    
    def _build_networks(self, hidden_dim: int, num_layers: int) -> None:
        """Build policy and value networks."""
        hidden_sizes = [hidden_dim] * num_layers
        
        if self.action_type == "discrete":
            self.policy_net = PolicyNetwork(
                self.obs_dim, self.action_dim, hidden_sizes
            ).to(self.device)
        else:
            self.policy_net = ContinuousPolicyNetwork(
                self.obs_dim, self.action_dim, hidden_sizes
            ).to(self.device)
        
        self.value_net = ValueNetwork(
            self.obs_dim, hidden_sizes
        ).to(self.device)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select an action using the current policy.
        
        Args:
            obs: Current observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, action_info)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.action_type == "discrete":
                action_probs = self.policy_net(obs_tensor)
                if deterministic:
                    action = torch.argmax(action_probs, dim=-1)
                else:
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                action_log_prob = torch.log(action_probs + 1e-8).gather(-1, action.unsqueeze(-1)).squeeze(-1)
            else:
                action_mean, action_std = self.policy_net(obs_tensor)
                if deterministic:
                    action = action_mean
                else:
                    action_dist = torch.distributions.Normal(action_mean, action_std)
                    action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            value = self.value_net(obs_tensor)
        
        action_np = action.cpu().numpy().squeeze()
        
        # Clip continuous actions
        if self.action_type == "continuous":
            action_np = self.clip_action(action_np, self.action_bounds)
        
        action_info = {
            'log_prob': action_log_prob.cpu().numpy().squeeze(),
            'value': value.cpu().numpy().squeeze(),
            'action_mean': action_mean.cpu().numpy().squeeze() if self.action_type == "continuous" else None,
            'action_std': action_std.cpu().numpy().squeeze() if self.action_type == "continuous" else None
        }
        
        return action_np, action_info
    
    def store_experience(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """
        Store experience in the buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        self.experience_buffer['observations'].append(obs)
        self.experience_buffer['actions'].append(action)
        self.experience_buffer['rewards'].append(reward)
        self.experience_buffer['values'].append(value)
        self.experience_buffer['log_probs'].append(log_prob)
        self.experience_buffer['dones'].append(done)
        
        self.step_count += 1
    
    def compute_gae_returns(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        Compute GAE returns and advantages.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of next state
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = []
        advantages = []
        
        # Add next value to values list
        values_with_next = values + [next_value]
        
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] * (1 - dones[t]) - values_with_next[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_with_next[t])
        
        return returns, advantages
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's policy and value networks.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.experience_buffer['observations']) < self.batch_size:
            return {}
        
        # Convert buffer to tensors
        obs_tensor = torch.FloatTensor(self.experience_buffer['observations']).to(self.device)
        action_tensor = torch.FloatTensor(self.experience_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.experience_buffer['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.experience_buffer['values']).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae_returns(
            self.experience_buffer['rewards'],
            self.experience_buffer['values'],
            self.experience_buffer['dones']
        )
        
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO updates
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divergences = []
        
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(obs_tensor))
            
            for start_idx in range(0, len(obs_tensor), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(obs_tensor))
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Policy update
                policy_loss, entropy, kl_div = self._update_policy(
                    batch_obs, batch_actions, batch_old_log_probs, batch_advantages
                )
                
                # Value update
                value_loss = self._update_value(batch_obs, batch_returns)
                
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                kl_divergences.append(kl_div)
        
        # Clear experience buffer
        self._clear_buffer()
        
        # Store training statistics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        avg_kl_div = np.mean(kl_divergences)
        
        self.policy_loss_history.append(avg_policy_loss)
        self.value_loss_history.append(avg_value_loss)
        self.entropy_history.append(avg_entropy)
        self.kl_divergence_history.append(avg_kl_div)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl_div,
            'total_loss': avg_policy_loss + self.value_loss_coef * avg_value_loss
        }
    
    def _update_policy(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Update policy network."""
        self.policy_optimizer.zero_grad()
        
        # Get current policy outputs
        if self.action_type == "discrete":
            action_probs = self.policy_net(obs)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions.long())
        else:
            action_mean, action_std = self.policy_net(obs)
            action_dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        # Compute policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy loss
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss
        
        # Backward pass
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        # Compute KL divergence
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean()
        
        return total_policy_loss.item(), entropy.item(), kl_div.item()
    
    def _update_value(self, obs: torch.Tensor, returns: torch.Tensor) -> float:
        """Update value network."""
        self.value_optimizer.zero_grad()
        
        values = self.value_net(obs)
        value_loss = F.mse_loss(values, returns)
        
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def _clear_buffer(self) -> None:
        """Clear the experience buffer."""
        for key in self.experience_buffer:
            self.experience_buffer[key] = []
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            'agent_id': self.agent_id,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'action_type': self.action_type,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size,
                'action_bounds': self.action_bounds
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved PPO agent checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.step_count = checkpoint['step_count']
        
        logger.info(f"Loaded PPO agent checkpoint from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = self.get_performance_stats()
        
        if self.policy_loss_history:
            stats.update({
                'avg_policy_loss': np.mean(self.policy_loss_history[-100:]),
                'avg_value_loss': np.mean(self.value_loss_history[-100:]),
                'avg_entropy': np.mean(self.entropy_history[-100:]),
                'avg_kl_divergence': np.mean(self.kl_divergence_history[-100:]),
                'buffer_size': len(self.experience_buffer['observations'])
            })
        
        return stats
