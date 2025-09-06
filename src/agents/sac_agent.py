"""
Soft Actor-Critic (SAC) Agent for Bipedal Locomotion

This module implements a SAC agent with automatic temperature tuning,
designed for continuous control tasks in bipedal locomotion.
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
from models.neural_networks.policy_networks import ContinuousPolicyNetwork
from models.neural_networks.value_networks import CriticNetwork

logger = logging.getLogger(__name__)


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic agent for bipedal locomotion.
    
    This agent implements the SAC algorithm with automatic temperature tuning,
    designed for continuous control tasks with high sample efficiency.
    """
    
    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        device: str = "cpu",
        # SAC-specific parameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        # Network architecture
        hidden_dim: int = 256,
        num_layers: int = 2,
        # Replay buffer
        buffer_size: int = 1000000,
        batch_size: int = 256,
        # Action bounds
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        # Training parameters
        update_frequency: int = 1,
        target_update_frequency: int = 1,
        **kwargs
    ):
        """
        Initialize SAC agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            action_type: Type of action space (should be "continuous" for SAC)
            device: Device to run computations on
            learning_rate: Learning rate for optimizers
            gamma: Discount factor for future rewards
            tau: Soft update coefficient for target networks
            alpha: Temperature parameter for entropy regularization
            automatic_entropy_tuning: Whether to automatically tune temperature
            target_entropy: Target entropy for automatic tuning
            hidden_dim: Hidden dimension for neural networks
            num_layers: Number of layers in neural networks
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            action_bounds: Action bounds for continuous actions
            update_frequency: Frequency of network updates
            target_update_frequency: Frequency of target network updates
        """
        super().__init__(agent_id, obs_dim, action_dim, action_type, device, **kwargs)
        
        # Ensure continuous action space for SAC
        if action_type != "continuous":
            logger.warning("SAC is designed for continuous actions. Converting to continuous.")
            self.action_type = "continuous"
        
        # SAC parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = target_entropy or -action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_bounds = action_bounds
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        # Initialize networks
        self._build_networks(hidden_dim, num_layers)
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)
        
        # Training statistics
        self.actor_loss_history = []
        self.critic1_loss_history = []
        self.critic2_loss_history = []
        self.alpha_history = []
        self.q1_values_history = []
        self.q2_values_history = []
        
        # Update counters
        self.update_count = 0
        
        logger.info(f"Initialized SAC agent {agent_id} with continuous actions")
    
    def _build_networks(self, hidden_dim: int, num_layers: int) -> None:
        """Build actor and critic networks."""
        hidden_sizes = [hidden_dim] * num_layers
        
        # Actor network (policy)
        self.actor = ContinuousPolicyNetwork(
            self.obs_dim, self.action_dim, hidden_sizes
        ).to(self.device)
        
        # Critic networks (Q-functions)
        self.critic1 = CriticNetwork(
            self.obs_dim, self.action_dim, hidden_sizes
        ).to(self.device)
        
        self.critic2 = CriticNetwork(
            self.obs_dim, self.action_dim, hidden_sizes
        ).to(self.device)
        
        # Target networks
        self.critic1_target = CriticNetwork(
            self.obs_dim, self.action_dim, hidden_sizes
        ).to(self.device)
        
        self.critic2_target = CriticNetwork(
            self.obs_dim, self.action_dim, hidden_sizes
        ).to(self.device)
        
        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
    def _build_optimizers(self) -> None:
        """Build optimizers for all networks."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
    
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
            action_mean, action_std = self.actor(obs_tensor)
            
            if deterministic:
                action = action_mean
            else:
                action_dist = torch.distributions.Normal(action_mean, action_std)
                action = action_dist.sample()
            
            # Apply tanh to bound actions
            action = torch.tanh(action)
            
            # Compute log probability
            log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
            # Correct for tanh transformation
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        action_np = action.cpu().numpy().squeeze()
        
        # Clip actions to bounds
        action_np = self.clip_action(action_np, self.action_bounds)
        
        action_info = {
            'log_prob': log_prob.cpu().numpy().squeeze(),
            'action_mean': action_mean.cpu().numpy().squeeze(),
            'action_std': action_std.cpu().numpy().squeeze(),
            'entropy': -log_prob.cpu().numpy().squeeze()
        }
        
        return action_np, action_info
    
    def store_experience(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in the replay buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.step_count += 1
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's networks using SAC algorithm.
        
        Args:
            batch: Dictionary containing training data (optional, uses replay buffer if not provided)
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch_data = self.replay_buffer.sample(self.batch_size)
        
        obs = torch.FloatTensor(batch_data['observations']).to(self.device)
        actions = torch.FloatTensor(batch_data['actions']).to(self.device)
        rewards = torch.FloatTensor(batch_data['rewards']).to(self.device)
        next_obs = torch.FloatTensor(batch_data['next_observations']).to(self.device)
        dones = torch.BoolTensor(batch_data['dones']).to(self.device)
        
        # Update networks
        metrics = {}
        
        if self.step_count % self.update_frequency == 0:
            # Update critics
            critic1_loss, critic2_loss = self._update_critics(obs, actions, rewards, next_obs, dones)
            metrics['critic1_loss'] = critic1_loss
            metrics['critic2_loss'] = critic2_loss
            
            # Update actor
            actor_loss, alpha_loss = self._update_actor(obs)
            metrics['actor_loss'] = actor_loss
            if alpha_loss is not None:
                metrics['alpha_loss'] = alpha_loss
            
            # Update target networks
            if self.step_count % self.target_update_frequency == 0:
                self._soft_update_target_networks()
            
            self.update_count += 1
        
        return metrics
    
    def _update_critics(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, float]:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_action_mean, next_action_std = self.actor(next_obs)
            next_action_dist = torch.distributions.Normal(next_action_mean, next_action_std)
            next_actions = next_action_dist.sample()
            next_actions = torch.tanh(next_actions)
            
            # Compute next Q-values
            next_q1 = self.critic1_target(next_obs, next_actions)
            next_q2 = self.critic2_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Compute target Q-values
            next_log_prob = next_action_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            next_log_prob -= torch.log(1 - next_actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            next_q = next_q - self.alpha * next_log_prob.squeeze()
            
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # Update critic 1
        current_q1 = self.critic1(obs, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        current_q2 = self.critic2(obs, actions)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Store Q-values for monitoring
        self.q1_values_history.append(current_q1.mean().item())
        self.q2_values_history.append(current_q2.mean().item())
        
        return critic1_loss.item(), critic2_loss.item()
    
    def _update_actor(self, obs: torch.Tensor) -> Tuple[float, Optional[float]]:
        """Update actor network."""
        # Sample actions from current policy
        action_mean, action_std = self.actor(obs)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        actions = action_dist.sample()
        actions = torch.tanh(actions)
        
        # Compute log probabilities
        log_prob = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        # Compute Q-values
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        q = torch.min(q1, q2)
        
        # Actor loss
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha if automatic tuning is enabled
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Store training statistics
        self.actor_loss_history.append(actor_loss.item())
        self.alpha_history.append(self.alpha)
        
        return actor_loss.item(), alpha_loss.item() if alpha_loss is not None else None
    
    def _soft_update_target_networks(self) -> None:
        """Soft update target networks."""
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            'agent_id': self.agent_id,
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'alpha': self.alpha,
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'action_type': self.action_type,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha': self.alpha,
                'automatic_entropy_tuning': self.automatic_entropy_tuning,
                'target_entropy': self.target_entropy,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'action_bounds': self.action_bounds
            }
        }
        
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved SAC agent checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        self.episode_count = checkpoint['episode_count']
        self.step_count = checkpoint['step_count']
        self.alpha = checkpoint['alpha']
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        logger.info(f"Loaded SAC agent checkpoint from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = self.get_performance_stats()
        
        if self.actor_loss_history:
            stats.update({
                'avg_actor_loss': np.mean(self.actor_loss_history[-100:]),
                'avg_critic1_loss': np.mean(self.critic1_loss_history[-100:]),
                'avg_critic2_loss': np.mean(self.critic2_loss_history[-100:]),
                'avg_q1_value': np.mean(self.q1_values_history[-100:]),
                'avg_q2_value': np.mean(self.q2_values_history[-100:]),
                'current_alpha': self.alpha,
                'buffer_size': len(self.replay_buffer),
                'update_count': self.update_count
            })
        
        return stats


class ReplayBuffer:
    """
    Experience replay buffer for SAC agent.
    """
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self) -> int:
        return self.size
