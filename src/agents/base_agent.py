"""
Base Agent Module for Multi-Agent Bipedal Locomotion

This module provides the abstract base class for all agents in the system,
defining the common interface and shared functionality.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    This class defines the common interface that all agents must implement,
    including action selection, learning, and communication capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            action_type: Type of action space ("discrete" or "continuous")
            device: Device to run computations on ("cpu" or "cuda")
            **kwargs: Additional agent-specific parameters
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.device = torch.device(device)
        
        # Agent state
        self.training = True
        self.episode_count = 0
        self.step_count = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {agent_id}")
    
    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, action_info) where action_info contains
            additional information like log probabilities, values, etc.
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's policy using a batch of experience.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics (losses, etc.)
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the agent's current state to a checkpoint file.
        
        Args:
            filepath: Path to save the checkpoint
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load the agent's state from a checkpoint file.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        pass
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set the agent's training mode.
        
        Args:
            training: Whether to set training mode (True) or evaluation mode (False)
        """
        self.training = training
        if hasattr(self, 'policy_net'):
            self.policy_net.train(training)
        if hasattr(self, 'value_net'):
            self.value_net.train(training)
        if hasattr(self, 'critic_net'):
            self.critic_net.train(training)
    
    def get_action_info(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get additional information about an action (log probs, values, etc.).
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Dictionary containing action information
        """
        return {}
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns from a sequence of rewards.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor
            
        Returns:
            List of discounted returns
        """
        returns = []
        running_return = 0
        
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def normalize_obs(self, obs: np.ndarray, obs_mean: Optional[np.ndarray] = None, 
                     obs_std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize observations using running statistics.
        
        Args:
            obs: Raw observations
            obs_mean: Mean for normalization (if None, no normalization)
            obs_std: Standard deviation for normalization (if None, no normalization)
            
        Returns:
            Normalized observations
        """
        if obs_mean is not None and obs_std is not None:
            return (obs - obs_mean) / (obs_std + 1e-8)
        return obs
    
    def clip_action(self, action: np.ndarray, action_bounds: Tuple[float, float]) -> np.ndarray:
        """
        Clip actions to valid bounds.
        
        Args:
            action: Raw action
            action_bounds: Tuple of (min_action, max_action)
            
        Returns:
            Clipped action
        """
        return np.clip(action, action_bounds[0], action_bounds[1])
    
    def log_episode(self, episode_reward: float, episode_length: int) -> None:
        """
        Log episode statistics.
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        
        if self.episode_count % 100 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            logger.info(f"Agent {self.agent_id} - Episode {self.episode_count}: "
                       f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.episode_rewards:
            return {}
        
        return {
            "episode_count": self.episode_count,
            "avg_reward": np.mean(self.episode_rewards),
            "avg_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "avg_length_100": np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths),
            "total_steps": self.step_count
        }
    
    def reset_episode(self) -> None:
        """
        Reset agent state for a new episode.
        """
        self.step_count = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, obs_dim={self.obs_dim}, action_dim={self.action_dim})"


class AgentConfig:
    """
    Configuration class for agent parameters.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        **kwargs
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        
        # Add any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
