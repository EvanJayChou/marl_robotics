"""
Communication Layer for Multi-Agent Bipedal Locomotion

This module implements inter-agent communication mechanisms including
message passing, attention-based communication, and coordination protocols.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MessageEncoder(nn.Module):
    """
    Encodes agent observations into communication messages.
    """
    
    def __init__(self, obs_dim: int, message_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation into message.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Message tensor of shape (batch_size, message_dim)
        """
        return self.encoder(obs)


class MessageDecoder(nn.Module):
    """
    Decodes received messages into useful information.
    """
    
    def __init__(self, message_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.message_dim = message_dim
        self.output_dim = output_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Decode message into output.
        
        Args:
            message: Message tensor of shape (batch_size, message_dim)
            
        Returns:
            Decoded tensor of shape (batch_size, output_dim)
        """
        return self.decoder(message)


class AttentionCommunication(nn.Module):
    """
    Attention-based communication mechanism for multi-agent systems.
    """
    
    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Message encoder
        self.message_encoder = MessageEncoder(obs_dim, message_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=message_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(message_dim, message_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(message_dim)
    
    def forward(
        self,
        obs_batch: torch.Tensor,
        agent_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform attention-based communication.
        
        Args:
            obs_batch: Observations from all agents (batch_size, num_agents, obs_dim)
            agent_masks: Optional masks for agent visibility (batch_size, num_agents, num_agents)
            
        Returns:
            Tuple of (attended_messages, attention_weights)
        """
        batch_size, num_agents, _ = obs_batch.shape
        
        # Encode observations into messages
        messages = self.message_encoder(obs_batch.view(-1, self.obs_dim))
        messages = messages.view(batch_size, num_agents, self.message_dim)
        
        # Apply attention
        attended_messages, attention_weights = self.attention(
            messages, messages, messages,
            attn_mask=agent_masks
        )
        
        # Residual connection and layer norm
        attended_messages = self.layer_norm(attended_messages + messages)
        attended_messages = self.output_proj(attended_messages)
        
        return attended_messages, attention_weights


class GraphCommunication(nn.Module):
    """
    Graph-based communication using Graph Neural Networks.
    """
    
    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_layers = num_layers
        
        # Node embedding
        self.node_embedding = nn.Linear(obs_dim, hidden_dim)
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Message projection
        self.message_proj = nn.Linear(hidden_dim, message_dim)
        
        # Activation
        self.activation = nn.ReLU()
    
    def forward(
        self,
        obs_batch: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform graph-based communication.
        
        Args:
            obs_batch: Observations from all agents (batch_size, num_agents, obs_dim)
            adjacency_matrix: Adjacency matrix for agent connections (batch_size, num_agents, num_agents)
            
        Returns:
            Communicated messages (batch_size, num_agents, message_dim)
        """
        batch_size, num_agents, _ = obs_batch.shape
        
        # Node embedding
        node_features = self.node_embedding(obs_batch)
        
        # Graph convolution
        for layer in self.gcn_layers:
            # Message passing
            messages = torch.bmm(adjacency_matrix, node_features)
            # Update node features
            node_features = self.activation(layer(messages))
        
        # Project to message dimension
        messages = self.message_proj(node_features)
        
        return messages


class CommunicationProtocol:
    """
    Protocol for managing inter-agent communication.
    """
    
    def __init__(
        self,
        num_agents: int,
        message_dim: int,
        communication_type: str = "attention",
        **kwargs
    ):
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.communication_type = communication_type
        
        # Initialize communication mechanism
        if communication_type == "attention":
            self.communication_net = AttentionCommunication(
                obs_dim=kwargs.get('obs_dim', 64),
                message_dim=message_dim,
                num_heads=kwargs.get('num_heads', 4),
                hidden_dim=kwargs.get('hidden_dim', 128)
            )
        elif communication_type == "graph":
            self.communication_net = GraphCommunication(
                obs_dim=kwargs.get('obs_dim', 64),
                message_dim=message_dim,
                num_layers=kwargs.get('num_layers', 2),
                hidden_dim=kwargs.get('hidden_dim', 128)
            )
        else:
            raise ValueError(f"Unknown communication type: {communication_type}")
        
        # Communication history
        self.message_history = []
        self.attention_history = []
    
    def communicate(
        self,
        obs_batch: torch.Tensor,
        agent_masks: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform inter-agent communication.
        
        Args:
            obs_batch: Observations from all agents
            agent_masks: Optional masks for agent visibility
            adjacency_matrix: Optional adjacency matrix for graph communication
            
        Returns:
            Tuple of (communicated_messages, communication_info)
        """
        if self.communication_type == "attention":
            messages, attention_weights = self.communication_net(obs_batch, agent_masks)
            communication_info = {
                "attention_weights": attention_weights,
                "message_norm": torch.norm(messages, dim=-1).mean()
            }
        elif self.communication_type == "graph":
            messages = self.communication_net(obs_batch, adjacency_matrix)
            communication_info = {
                "message_norm": torch.norm(messages, dim=-1).mean()
            }
        else:
            raise ValueError(f"Unknown communication type: {self.communication_type}")
        
        # Store communication history
        self.message_history.append(messages.detach().cpu())
        if "attention_weights" in communication_info:
            self.attention_history.append(attention_weights.detach().cpu())
        
        return messages, communication_info
    
    def get_communication_stats(self) -> Dict[str, float]:
        """
        Get communication statistics.
        
        Returns:
            Dictionary of communication metrics
        """
        if not self.message_history:
            return {}
        
        recent_messages = torch.cat(self.message_history[-10:], dim=0)
        message_norm = torch.norm(recent_messages, dim=-1).mean().item()
        
        stats = {
            "message_norm": message_norm,
            "communication_steps": len(self.message_history)
        }
        
        if self.attention_history:
            recent_attention = torch.cat(self.attention_history[-10:], dim=0)
            attention_entropy = -torch.sum(
                recent_attention * torch.log(recent_attention + 1e-8), dim=-1
            ).mean().item()
            stats["attention_entropy"] = attention_entropy
        
        return stats
    
    def reset_history(self) -> None:
        """Reset communication history."""
        self.message_history = []
        self.attention_history = []


class CoordinationMechanism:
    """
    High-level coordination mechanism for multi-agent systems.
    """
    
    def __init__(
        self,
        num_agents: int,
        coordination_type: str = "centralized",
        **kwargs
    ):
        self.num_agents = num_agents
        self.coordination_type = coordination_type
        
        if coordination_type == "centralized":
            self.coordinator = CentralizedCoordinator(num_agents, **kwargs)
        elif coordination_type == "decentralized":
            self.coordinator = DecentralizedCoordinator(num_agents, **kwargs)
        elif coordination_type == "hierarchical":
            self.coordinator = HierarchicalCoordinator(num_agents, **kwargs)
        else:
            raise ValueError(f"Unknown coordination type: {coordination_type}")
    
    def coordinate(
        self,
        agent_observations: List[np.ndarray],
        agent_actions: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Coordinate agent actions.
        
        Args:
            agent_observations: List of observations from each agent
            agent_actions: List of proposed actions from each agent
            
        Returns:
            Coordination information and potentially modified actions
        """
        return self.coordinator.coordinate(agent_observations, agent_actions, **kwargs)


class CentralizedCoordinator:
    """Centralized coordination mechanism."""
    
    def __init__(self, num_agents: int, **kwargs):
        self.num_agents = num_agents
    
    def coordinate(
        self,
        agent_observations: List[np.ndarray],
        agent_actions: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """Centralized coordination logic."""
        # Simple centralized coordination - can be extended
        return {
            "coordinated_actions": agent_actions,
            "coordination_info": {"type": "centralized"}
        }


class DecentralizedCoordinator:
    """Decentralized coordination mechanism."""
    
    def __init__(self, num_agents: int, **kwargs):
        self.num_agents = num_agents
    
    def coordinate(
        self,
        agent_observations: List[np.ndarray],
        agent_actions: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """Decentralized coordination logic."""
        # Simple decentralized coordination - can be extended
        return {
            "coordinated_actions": agent_actions,
            "coordination_info": {"type": "decentralized"}
        }


class HierarchicalCoordinator:
    """Hierarchical coordination mechanism."""
    
    def __init__(self, num_agents: int, **kwargs):
        self.num_agents = num_agents
    
    def coordinate(
        self,
        agent_observations: List[np.ndarray],
        agent_actions: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """Hierarchical coordination logic."""
        # Simple hierarchical coordination - can be extended
        return {
            "coordinated_actions": agent_actions,
            "coordination_info": {"type": "hierarchical"}
        }
