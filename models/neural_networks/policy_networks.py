import torch
import torch.nn as nn
import torch.nn.functional as F

# === BASIC POLICY NETWORK ===
class PolicyNetwork(nn.Module):
    """
    Basic policy network for discrete action spaces.
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
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        x = self.net(state)
        action_probs = F.softmax(x, dim=-1)
        return action_probs

# === CONTINUOUS POLICY NETWORK ===
class ContinuousPolicyNetwork(nn.Module):
    """
    Policy network for continuous action spaces with learnable standard deviation.
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
        
        self.mean_net = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.mean_net(state)
        std = self.log_std.exp()
        return mean, std

# === SHARED POLICY NETWORK ===
class SharedPolicyNetwork(nn.Module):
    """
    Shared policy network that can be used across multiple agents.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128, action_type="discrete"):
        super().__init__()
        self.action_type = action_type
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if action_type == "discrete":
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        features = self.shared_encoder(obs)
        
        if self.action_type == "discrete":
            action_logits = self.action_head(features)
            return action_logits
        else:
            action_mean = self.action_head(features)
            action_std = self.log_std.exp().expand_as(action_mean)
            return action_mean, action_std