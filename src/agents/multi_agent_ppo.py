"""
Multi-Agent PPO System for Bipedal Locomotion

This module implements a multi-agent PPO system with communication and coordination
mechanisms for multiple bipedal robots working together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
from collections import defaultdict, deque

from .base_agent import BaseAgent, AgentConfig
from .ppo_agent import PPOAgent
from .communication_layer import CommunicationProtocol, CoordinationMechanism
from .attention_agent import AttentionAgent
from models.neural_networks.attention_models import SharedAttentionEncoder, PolicyHead, ValueHead, CentralizedCritic
from models.neural_networks.value_networks import CentralizedValueNetwork

logger = logging.getLogger(__name__)


class MultiAgentPPO(BaseAgent):
    """
    Multi-Agent PPO system for coordinated bipedal locomotion.
    
    This system manages multiple PPO agents with communication and coordination
    mechanisms, supporting both centralized and decentralized training approaches.
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        device: str = "cpu",
        # Multi-agent specific parameters
        num_agents: int = 2,
        communication_type: str = "attention",
        coordination_type: str = "centralized",
        use_centralized_critic: bool = True,
        use_shared_policy: bool = False,
        # PPO parameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        # Communication parameters
        message_dim: int = 64,
        num_attention_heads: int = 4,
        # Network architecture
        hidden_dim: int = 128,
        num_layers: int = 2,
        # Action bounds
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        **kwargs
    ):
        """
        Initialize Multi-Agent PPO system.
        
        Args:
            agent_ids: List of unique identifiers for each agent
            obs_dim: Dimension of observation space per agent
            action_dim: Dimension of action space per agent
            action_type: Type of action space ("discrete" or "continuous")
            device: Device to run computations on
            num_agents: Number of agents in the system
            communication_type: Type of communication ("attention", "graph", "none")
            coordination_type: Type of coordination ("centralized", "decentralized", "hierarchical")
            use_centralized_critic: Whether to use centralized critic
            use_shared_policy: Whether to use shared policy across agents
            learning_rate: Learning rate for optimizers
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy loss
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs
            batch_size: Batch size for training
            message_dim: Dimension of communication messages
            num_attention_heads: Number of attention heads for communication
            hidden_dim: Hidden dimension for neural networks
            num_layers: Number of layers in neural networks
            action_bounds: Action bounds for continuous actions
        """
        # Use the first agent ID as the main agent ID for base class
        super().__init__(agent_ids[0], obs_dim, action_dim, action_type, device, **kwargs)
        
        self.agent_ids = agent_ids
        self.num_agents = num_agents
        self.communication_type = communication_type
        self.coordination_type = coordination_type
        self.use_centralized_critic = use_centralized_critic
        self.use_shared_policy = use_shared_policy
        
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
        
        # Initialize communication and coordination
        self._build_communication_system(message_dim, num_attention_heads)
        self._build_coordination_system()
        
        # Initialize networks
        self._build_networks(hidden_dim, num_layers)
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Experience buffers for each agent
        self.experience_buffers = {
            agent_id: {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': [],
                'messages': [],
                'attention_weights': []
            } for agent_id in agent_ids
        }
        
        # Training statistics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.kl_divergence_history = []
        self.communication_loss_history = []
        
        logger.info(f"Initialized Multi-Agent PPO with {num_agents} agents")
    
    def _build_communication_system(self, message_dim: int, num_attention_heads: int) -> None:
        """Build communication system."""
        if self.communication_type != "none":
            self.communication_protocol = CommunicationProtocol(
                num_agents=self.num_agents,
                message_dim=message_dim,
                communication_type=self.communication_type,
                obs_dim=self.obs_dim,
                num_heads=num_attention_heads,
                hidden_dim=128
            )
        else:
            self.communication_protocol = None
    
    def _build_coordination_system(self) -> None:
        """Build coordination system."""
        self.coordination_mechanism = CoordinationMechanism(
            num_agents=self.num_agents,
            coordination_type=self.coordination_type
        )
    
    def _build_networks(self, hidden_dim: int, num_layers: int) -> None:
        """Build policy and value networks."""
        hidden_sizes = [hidden_dim] * num_layers
        
        if self.use_shared_policy:
            # Shared policy network
            if self.action_type == "discrete":
                from models.neural_networks.policy_networks import PolicyNetwork
                self.policy_net = PolicyNetwork(
                    self.obs_dim, self.action_dim, hidden_sizes
                ).to(self.device)
            else:
                from models.neural_networks.policy_networks import ContinuousPolicyNetwork
                self.policy_net = ContinuousPolicyNetwork(
                    self.obs_dim, self.action_dim, hidden_sizes
                ).to(self.device)
        else:
            # Individual policy networks for each agent
            self.policy_nets = {}
            for agent_id in self.agent_ids:
                if self.action_type == "discrete":
                    from models.neural_networks.policy_networks import PolicyNetwork
                    self.policy_nets[agent_id] = PolicyNetwork(
                        self.obs_dim, self.action_dim, hidden_sizes
                    ).to(self.device)
                else:
                    from models.neural_networks.policy_networks import ContinuousPolicyNetwork
                    self.policy_nets[agent_id] = ContinuousPolicyNetwork(
                        self.obs_dim, self.action_dim, hidden_sizes
                    ).to(self.device)
        
        # Value networks
        if self.use_centralized_critic:
            # Centralized value network
            self.value_net = CentralizedValueNetwork(
                self.obs_dim, self.num_agents, hidden_sizes
            ).to(self.device)
        else:
            # Individual value networks
            self.value_nets = {}
            for agent_id in self.agent_ids:
                from models.neural_networks.value_networks import ValueNetwork
                self.value_nets[agent_id] = ValueNetwork(
                    self.obs_dim, hidden_sizes
                ).to(self.device)
    
    def _build_optimizers(self) -> None:
        """Build optimizers for all networks."""
        if self.use_shared_policy:
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            self.policy_optimizers = {
                agent_id: optim.Adam(self.policy_nets[agent_id].parameters(), lr=self.learning_rate)
                for agent_id in self.agent_ids
            }
        
        if self.use_centralized_critic:
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        else:
            self.value_optimizers = {
                agent_id: optim.Adam(self.value_nets[agent_id].parameters(), lr=self.learning_rate)
                for agent_id in self.agent_ids
            }
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select actions for all agents.
        
        Args:
            obs: Observations from all agents (num_agents, obs_dim)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (actions, action_info)
        """
        if obs.shape[0] != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} observations, got {obs.shape[0]}")
        
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions = []
        action_info = {}
        
        # Communication phase
        if self.communication_protocol is not None:
            obs_batch = obs_tensor.unsqueeze(0)  # Add batch dimension
            messages, comm_info = self.communication_protocol.communicate(obs_batch)
            messages = messages.squeeze(0)  # Remove batch dimension
            action_info['communication'] = comm_info
        else:
            messages = None
        
        # Action selection phase
        for i, agent_id in enumerate(self.agent_ids):
            agent_obs = obs_tensor[i]
            
            # Get policy network
            if self.use_shared_policy:
                policy_net = self.policy_net
            else:
                policy_net = self.policy_nets[agent_id]
            
            with torch.no_grad():
                if self.action_type == "discrete":
                    action_probs = policy_net(agent_obs.unsqueeze(0))
                    if deterministic:
                        action = torch.argmax(action_probs, dim=-1)
                    else:
                        action_dist = torch.distributions.Categorical(action_probs)
                        action = action_dist.sample()
                    action_log_prob = torch.log(action_probs + 1e-8).gather(-1, action.unsqueeze(-1)).squeeze(-1)
                else:
                    action_mean, action_std = policy_net(agent_obs.unsqueeze(0))
                    if deterministic:
                        action = action_mean
                    else:
                        action_dist = torch.distributions.Normal(action_mean, action_std)
                        action = action_dist.sample()
                    action_log_prob = action_dist.log_prob(action).sum(dim=-1)
                
                # Get value estimate
                if self.use_centralized_critic:
                    all_obs = obs_tensor.unsqueeze(0)
                    value = self.value_net(all_obs)
                    value = value[i]  # Get value for this agent
                else:
                    value = self.value_nets[agent_id](agent_obs.unsqueeze(0))
            
            action_np = action.cpu().numpy().squeeze()
            
            # Clip continuous actions
            if self.action_type == "continuous":
                action_np = self.clip_action(action_np, self.action_bounds)
            
            actions.append(action_np)
            action_info[agent_id] = {
                'log_prob': action_log_prob.cpu().numpy().squeeze(),
                'value': value.cpu().numpy().squeeze(),
                'action_mean': action_mean.cpu().numpy().squeeze() if self.action_type == "continuous" else None,
                'action_std': action_std.cpu().numpy().squeeze() if self.action_type == "continuous" else None
            }
        
        # Coordination phase
        coordination_result = self.coordination_mechanism.coordinate(
            obs.tolist(), actions
        )
        action_info['coordination'] = coordination_result
        
        return np.array(actions), action_info
    
    def store_experience(
        self,
        agent_id: str,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        message: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None
    ) -> None:
        """
        Store experience for a specific agent.
        
        Args:
            agent_id: ID of the agent
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
            message: Communication message (optional)
            attention_weights: Attention weights (optional)
        """
        if agent_id not in self.experience_buffers:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        self.experience_buffers[agent_id]['observations'].append(obs)
        self.experience_buffers[agent_id]['actions'].append(action)
        self.experience_buffers[agent_id]['rewards'].append(reward)
        self.experience_buffers[agent_id]['values'].append(value)
        self.experience_buffers[agent_id]['log_probs'].append(log_prob)
        self.experience_buffers[agent_id]['dones'].append(done)
        
        if message is not None:
            self.experience_buffers[agent_id]['messages'].append(message)
        if attention_weights is not None:
            self.experience_buffers[agent_id]['attention_weights'].append(attention_weights)
        
        self.step_count += 1
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update all agents' networks using multi-agent PPO.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics
        """
        # Check if we have enough experience
        min_buffer_size = min(
            len(self.experience_buffers[agent_id]['observations'])
            for agent_id in self.agent_ids
        )
        
        if min_buffer_size < self.batch_size:
            return {}
        
        # Update each agent
        total_metrics = defaultdict(list)
        
        for agent_id in self.agent_ids:
            agent_metrics = self._update_agent(agent_id)
            for key, value in agent_metrics.items():
                total_metrics[key].append(value)
        
        # Average metrics across agents
        avg_metrics = {
            key: np.mean(values) for key, values in total_metrics.items()
        }
        
        # Clear experience buffers
        self._clear_buffers()
        
        # Store training statistics
        if avg_metrics:
            self.policy_loss_history.append(avg_metrics.get('policy_loss', 0))
            self.value_loss_history.append(avg_metrics.get('value_loss', 0))
            self.entropy_history.append(avg_metrics.get('entropy', 0))
            self.kl_divergence_history.append(avg_metrics.get('kl_divergence', 0))
        
        return avg_metrics
    
    def _update_agent(self, agent_id: str) -> Dict[str, float]:
        """Update a specific agent's networks."""
        buffer = self.experience_buffers[agent_id]
        
        # Convert buffer to tensors
        obs_tensor = torch.FloatTensor(buffer['observations']).to(self.device)
        action_tensor = torch.FloatTensor(buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer['log_probs']).to(self.device)
        old_values = torch.FloatTensor(buffer['values']).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae_returns(
            buffer['rewards'],
            buffer['values'],
            buffer['dones']
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
                    agent_id, batch_obs, batch_actions, batch_old_log_probs, batch_advantages
                )
                
                # Value update
                value_loss = self._update_value(agent_id, batch_obs, batch_returns)
                
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                kl_divergences.append(kl_div)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divergences)
        }
    
    def _compute_gae_returns(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE returns and advantages."""
        returns = []
        advantages = []
        
        values_with_next = values + [next_value]
        
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] * (1 - dones[t]) - values_with_next[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_with_next[t])
        
        return returns, advantages
    
    def _update_policy(
        self,
        agent_id: str,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Update policy network for a specific agent."""
        # Get policy network and optimizer
        if self.use_shared_policy:
            policy_net = self.policy_net
            optimizer = self.policy_optimizer
        else:
            policy_net = self.policy_nets[agent_id]
            optimizer = self.policy_optimizers[agent_id]
        
        optimizer.zero_grad()
        
        # Get current policy outputs
        if self.action_type == "discrete":
            action_probs = policy_net(obs)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions.long())
        else:
            action_mean, action_std = policy_net(obs)
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
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.max_grad_norm)
        optimizer.step()
        
        # Compute KL divergence
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean()
        
        return total_policy_loss.item(), entropy.item(), kl_div.item()
    
    def _update_value(
        self,
        agent_id: str,
        obs: torch.Tensor,
        returns: torch.Tensor
    ) -> float:
        """Update value network for a specific agent."""
        if self.use_centralized_critic:
            # For centralized critic, we need all observations
            # This is a simplified version - in practice, you'd need to handle this properly
            value_net = self.value_net
            optimizer = self.value_optimizer
            # Note: This is a simplified implementation
            # In practice, you'd need to handle the centralized critic properly
            values = value_net(obs.unsqueeze(0))  # This won't work as expected
        else:
            value_net = self.value_nets[agent_id]
            optimizer = self.value_optimizers[agent_id]
            values = value_net(obs)
        
        optimizer.zero_grad()
        value_loss = F.mse_loss(values, returns)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
        optimizer.step()
        
        return value_loss.item()
    
    def _clear_buffers(self) -> None:
        """Clear all experience buffers."""
        for agent_id in self.agent_ids:
            for key in self.experience_buffers[agent_id]:
                self.experience_buffers[agent_id][key] = []
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save multi-agent system checkpoint."""
        checkpoint = {
            'agent_ids': self.agent_ids,
            'num_agents': self.num_agents,
            'communication_type': self.communication_type,
            'coordination_type': self.coordination_type,
            'use_centralized_critic': self.use_centralized_critic,
            'use_shared_policy': self.use_shared_policy,
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
        
        # Save networks
        if self.use_shared_policy:
            checkpoint['policy_net_state_dict'] = self.policy_net.state_dict()
            checkpoint['policy_optimizer_state_dict'] = self.policy_optimizer.state_dict()
        else:
            checkpoint['policy_nets_state_dict'] = {
                agent_id: self.policy_nets[agent_id].state_dict()
                for agent_id in self.agent_ids
            }
            checkpoint['policy_optimizers_state_dict'] = {
                agent_id: self.policy_optimizers[agent_id].state_dict()
                for agent_id in self.agent_ids
            }
        
        if self.use_centralized_critic:
            checkpoint['value_net_state_dict'] = self.value_net.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        else:
            checkpoint['value_nets_state_dict'] = {
                agent_id: self.value_nets[agent_id].state_dict()
                for agent_id in self.agent_ids
            }
            checkpoint['value_optimizers_state_dict'] = {
                agent_id: self.value_optimizers[agent_id].state_dict()
                for agent_id in self.agent_ids
            }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved Multi-Agent PPO checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load multi-agent system checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load networks
        if self.use_shared_policy:
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        else:
            for agent_id in self.agent_ids:
                self.policy_nets[agent_id].load_state_dict(
                    checkpoint['policy_nets_state_dict'][agent_id]
                )
                self.policy_optimizers[agent_id].load_state_dict(
                    checkpoint['policy_optimizers_state_dict'][agent_id]
                )
        
        if self.use_centralized_critic:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        else:
            for agent_id in self.agent_ids:
                self.value_nets[agent_id].load_state_dict(
                    checkpoint['value_nets_state_dict'][agent_id]
                )
                self.value_optimizers[agent_id].load_state_dict(
                    checkpoint['value_optimizers_state_dict'][agent_id]
                )
        
        self.episode_count = checkpoint['episode_count']
        self.step_count = checkpoint['step_count']
        
        logger.info(f"Loaded Multi-Agent PPO checkpoint from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for the multi-agent system."""
        stats = self.get_performance_stats()
        
        if self.policy_loss_history:
            stats.update({
                'avg_policy_loss': np.mean(self.policy_loss_history[-100:]),
                'avg_value_loss': np.mean(self.value_loss_history[-100:]),
                'avg_entropy': np.mean(self.entropy_history[-100:]),
                'avg_kl_divergence': np.mean(self.kl_divergence_history[-100:]),
                'num_agents': self.num_agents,
                'communication_type': self.communication_type,
                'coordination_type': self.coordination_type
            })
        
        # Add communication statistics if available
        if self.communication_protocol is not None:
            comm_stats = self.communication_protocol.get_communication_stats()
            stats.update(comm_stats)
        
        return stats
