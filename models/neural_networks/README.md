# Neural Networks Module

This module contains pure neural network architectures and components for multi-agent reinforcement learning. All components are designed to be modular and reusable.

## Structure

The neural networks are organized into the following categories:

### Policy Networks (`policy_networks.py`)
- **PolicyNetwork**: Basic policy network for discrete action spaces
- **ContinuousPolicyNetwork**: Policy network for continuous action spaces with learnable standard deviation
- **SharedPolicyNetwork**: Shared policy network that can be used across multiple agents

### Value Networks (`value_networks.py`)
- **ValueNetwork**: Basic value network for state value estimation
- **CentralizedValueNetwork**: Centralized value network for CTDE approaches
- **DualValueNetwork**: Network that outputs both state value and advantage function
- **CriticNetwork**: Critic network for actor-critic methods

### Attention Models (`attention_models.py`)
- **SharedAttentionEncoder**: Transformer-based encoder for processing observation sequences
- **PolicyHead**: Policy head for attention-based architectures
- **ValueHead**: Value head for attention-based architectures
- **CentralizedCritic**: Centralized critic for multi-agent scenarios

### Shared Networks (`shared_networks.py`)
- **SharedEncoder**: Basic shared encoder for parameter sharing across agents
- **SharedValueNetwork**: Shared value network for centralized critics
- **SharedAttention**: Multi-head attention mechanism for agent interactions
- **SharedRNN**: RNN for temporal sequence modeling (GRU/LSTM)
- **SharedTransformer**: Transformer encoder for sequence processing
- **SharedConvEncoder**: Convolutional encoder for spatial observations

## Usage

### Basic Usage

```python
from models.neural_networks import PolicyNetwork, ValueNetwork

# Create networks
policy_net = PolicyNetwork(state_dim=64, action_dim=4)
value_net = ValueNetwork(state_dim=64)

# Forward pass
state = torch.randn(32, 64)
action_probs = policy_net(state)
value = value_net(state)
```

### Attention-Based Components

```python
from models.neural_networks import SharedAttentionEncoder, PolicyHead, ValueHead

# Create attention components
encoder = SharedAttentionEncoder(obs_dim=32, hidden_dim=128)
policy_head = PolicyHead(hidden_dim=128, action_dim=6)
value_head = ValueHead(hidden_dim=128)

# Process observations
obs_batch = torch.randn(16, 10, 32)  # batch_size, seq_len, obs_dim
encoded = encoder(obs_batch)
policy_output = policy_head(encoded)
value_output = value_head(encoded)
```

### Creating Custom Agents

```python
import torch.nn as nn
from models.neural_networks import SharedAttentionEncoder, PolicyHead, ValueHead

class CustomAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.encoder = SharedAttentionEncoder(obs_dim, hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, action_dim)
        self.value_head = ValueHead(hidden_dim)
    
    def forward(self, obs_batch):
        encoded = self.encoder(obs_batch)
        policy_output = self.policy_head(encoded)
        value_output = self.value_head(encoded)
        return {"policy": policy_output, "value": value_output}
```

## Design Principles

1. **Modularity**: Each component is self-contained and can be used independently
2. **Reusability**: Components can be combined in different ways to create various architectures
3. **Flexibility**: Support for both discrete and continuous action spaces
4. **Scalability**: Designed to work with multiple agents and centralized training
5. **Extensibility**: Easy to add new components or modify existing ones

## Agent Implementations

For complete agent implementations that use these neural network components, see the `src/agents` module. The agents module contains:

- **AttentionAgent**: Complete attention-based agent implementation
- Other agent types (to be implemented)

## Examples

See `examples/neural_networks_usage.py` for comprehensive usage examples.

## Contributing

When adding new neural network components:

1. Follow the existing naming conventions
2. Add comprehensive docstrings
3. Include type hints where appropriate
4. Update the `__init__.py` file to export new components
5. Add usage examples to the examples file 