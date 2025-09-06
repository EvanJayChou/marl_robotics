"""
Multi-Agent Bipedal Locomotion Agents Module

This module provides various agent implementations for multi-agent bipedal locomotion,
including single-agent algorithms (PPO, SAC) and multi-agent coordination systems.
"""

from .base_agent import BaseAgent, AgentConfig
from .attention_agent import AttentionAgent
from .communication_layer import (
    MessageEncoder, 
    MessageDecoder, 
    AttentionCommunication, 
    GraphCommunication,
    CommunicationProtocol,
    CoordinationMechanism,
    CentralizedCoordinator,
    DecentralizedCoordinator,
    HierarchicalCoordinator
)
from .ppo_agent import PPOAgent
from .sac_agent import SACAgent, ReplayBuffer
from .multi_agent_ppo import MultiAgentPPO

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    
    # Single agent algorithms
    "AttentionAgent",
    "PPOAgent", 
    "SACAgent",
    "ReplayBuffer",
    
    # Multi-agent systems
    "MultiAgentPPO",
    
    # Communication and coordination
    "MessageEncoder",
    "MessageDecoder", 
    "AttentionCommunication",
    "GraphCommunication",
    "CommunicationProtocol",
    "CoordinationMechanism",
    "CentralizedCoordinator",
    "DecentralizedCoordinator", 
    "HierarchicalCoordinator"
]
