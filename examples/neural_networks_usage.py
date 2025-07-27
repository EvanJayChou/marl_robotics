"""
Example demonstrating the usage of refactored neural networks and agents.

This example shows how to:
1. Use pure neural network components from models.neural_networks
2. Use agent implementations from src.agents
3. Create custom agents by combining neural network components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import neural network components
from models.neural_networks import (
    PolicyNetwork,
    ContinuousPolicyNetwork,
    ValueNetwork,
    CentralizedValueNetwork,
    SharedAttentionEncoder,
    PolicyHead,
    ValueHead,
    CentralizedCritic
)

# Import agent implementations
from src.agents import AttentionAgent

def example_basic_networks():
    """Example of using basic neural network components."""
    print("=== Basic Neural Network Components ===")
    
    # Policy network for discrete actions
    policy_net = PolicyNetwork(state_dim=64, action_dim=4)
    state = torch.randn(32, 64)  # batch_size=32, state_dim=64
    action_probs = policy_net(state)
    print(f"Policy network output shape: {action_probs.shape}")
    
    # Value network
    value_net = ValueNetwork(state_dim=64)
    value = value_net(state)
    print(f"Value network output shape: {value.shape}")
    
    # Continuous policy network
    cont_policy_net = ContinuousPolicyNetwork(state_dim=64, action_dim=8)
    action_mean, action_std = cont_policy_net(state)
    print(f"Continuous policy mean shape: {action_mean.shape}")
    print(f"Continuous policy std shape: {action_std.shape}")

def example_attention_components():
    """Example of using attention-based neural network components."""
    print("\n=== Attention-Based Components ===")
    
    # Shared attention encoder
    attention_encoder = SharedAttentionEncoder(obs_dim=32, hidden_dim=128)
    obs_batch = torch.randn(16, 10, 32)  # batch_size=16, seq_len=10, obs_dim=32
    encoded = attention_encoder(obs_batch)
    print(f"Attention encoder output shape: {encoded.shape}")
    
    # Policy head
    policy_head = PolicyHead(hidden_dim=128, action_dim=6)
    policy_output = policy_head(encoded)
    print(f"Policy head output shape: {policy_output.shape}")
    
    # Value head
    value_head = ValueHead(hidden_dim=128)
    value_output = value_head(encoded)
    print(f"Value head output shape: {value_output.shape}")
    
    # Centralized critic
    centralized_critic = CentralizedCritic(num_agents=4, hidden_dim=128)
    all_agent_features = torch.randn(16, 4, 128)  # batch_size=16, num_agents=4, hidden_dim=128
    central_value = centralized_critic(all_agent_features)
    print(f"Centralized critic output shape: {central_value.shape}")

def example_attention_agent():
    """Example of using the attention-based agent."""
    print("\n=== Attention-Based Agent ===")
    
    # Create attention agent
    agent = AttentionAgent(
        obs_dim=32,
        action_dim=6,
        num_agents=4,
        hidden_dim=128,
        action_type="discrete",
        use_centralized_critic=True
    )
    
    # Forward pass
    obs_batch = torch.randn(16, 10, 32)  # batch_size=16, seq_len=10, obs_dim=32
    outputs = agent(obs_batch)
    print(f"Agent outputs keys: {outputs.keys()}")
    print(f"Policy output shape: {outputs['policy'].shape}")
    print(f"Value output shape: {outputs['value'].shape}")
    print(f"Central value output shape: {outputs['central_value'].shape}")
    
    # Get action
    action, action_log_prob, value = agent.get_action(obs_batch)
    print(f"Action shape: {action.shape}")
    print(f"Action log prob shape: {action_log_prob.shape}")
    print(f"Value shape: {value.shape}")

def example_custom_agent():
    """Example of creating a custom agent by combining neural network components."""
    print("\n=== Custom Agent Example ===")
    
    class CustomAgent(nn.Module):
        """Custom agent that combines different neural network components."""
        
        def __init__(self, obs_dim, action_dim, hidden_dim=128):
            super().__init__()
            
            # Use shared attention encoder
            self.encoder = SharedAttentionEncoder(obs_dim, hidden_dim)
            
            # Use policy and value heads
            self.policy_head = PolicyHead(hidden_dim, action_dim, action_type="discrete")
            self.value_head = ValueHead(hidden_dim)
            
            # Add a custom component
            self.custom_layer = nn.Linear(hidden_dim, hidden_dim)
            
        def forward(self, obs_batch):
            # Encode observations
            encoded = self.encoder(obs_batch)
            
            # Apply custom processing
            processed = F.relu(self.custom_layer(encoded))
            
            # Get policy and value outputs
            policy_output = self.policy_head(processed)
            value_output = self.value_head(processed)
            
            return {
                "policy": policy_output,
                "value": value_output,
                "features": processed
            }
    
    # Create and use custom agent
    custom_agent = CustomAgent(obs_dim=32, action_dim=6)
    obs_batch = torch.randn(16, 10, 32)
    outputs = custom_agent(obs_batch)
    print(f"Custom agent outputs keys: {outputs.keys()}")
    print(f"Features shape: {outputs['features'].shape}")

def example_centralized_training():
    """Example of centralized training setup."""
    print("\n=== Centralized Training Setup ===")
    
    # Centralized value network for global state evaluation
    centralized_value_net = CentralizedValueNetwork(obs_dim=32, num_agents=4)
    global_obs = torch.randn(16, 4, 32)  # batch_size=16, num_agents=4, obs_dim=32
    global_value = centralized_value_net(global_obs)
    print(f"Global value shape: {global_value.shape}")
    
    # Multiple agents with shared components
    agents = []
    for i in range(4):
        agent = AttentionAgent(
            obs_dim=32,
            action_dim=6,
            num_agents=4,
            hidden_dim=128,
            use_centralized_critic=True
        )
        agents.append(agent)
    
    print(f"Created {len(agents)} agents with shared attention components")

if __name__ == "__main__":
    print("Neural Networks and Agents Usage Examples")
    print("=" * 50)
    
    example_basic_networks()
    example_attention_components()
    example_attention_agent()
    example_custom_agent()
    example_centralized_training()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!") 