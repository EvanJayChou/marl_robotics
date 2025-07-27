import torch
import torch.nn as nn
import torch.nn.functional as F

# === BASIC VALUE NETWORK ===
class ValueNetwork(nn.Module):
    """
    Basic value network for state value estimation.
    """
    def __init__(self, state_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.net(state).squeeze(-1)

# === CENTRALIZED VALUE NETWORK ===
class CentralizedValueNetwork(nn.Module):
    """
    Centralized value network that takes observations from all agents.
    Used in centralized training with decentralized execution (CTDE) approaches.
    """
    def __init__(self, obs_dim, num_agents, hidden_sizes=(256, 128)):
        super().__init__()
        input_dim = obs_dim * num_agents
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs_all_agents):
        batch_size = obs_all_agents.shape[0]
        x = obs_all_agents.view(batch_size, -1)
        return self.net(x).squeeze(-1)

# === DUAL VALUE NETWORK ===
class DualValueNetwork(nn.Module):
    """
    Network that outputs both state value and advantage function.
    Used in algorithms like Dueling DQN.
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Value stream
        self.value_stream = nn.Linear(prev_dim, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        features = self.shared_features(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
    
    def shared_features(self, state):
        # This would be implemented based on the specific architecture
        # For now, we'll use a simple approach
        return state

# === CRITIC NETWORK ===
class CriticNetwork(nn.Module):
    """
    Critic network for actor-critic methods.
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)