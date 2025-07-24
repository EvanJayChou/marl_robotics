import torch
import torch.nn as nn
import torch.nn.functional as F

# === SHARED ATTENTION ENCODER ===
class SharedAttentionEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, obs_batch):
        x = self.obs_proj(obs_batch)
        x = self.transformer_encoder(x)
        return x

# === POLICY HEAD ===
class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, action_type="discrete"):
        super().__init__()
        self.action_type=action_type
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)


        if action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        action_logits = self.fc2(x)

        if self.action_type == "discrete":
            return action_logits
        else:
            return action_logits, self.log_std.exp().expand_as(action_logits)

# === VALUE HEAD ===
class ValueHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        values = self.fc2(x).squeeze(-1)
        return values

class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, hidden_dim):
        super().__init__()
        input_dim = num_agents * hidden_dim
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, encoded_all_agents):
        x = encoded_all_agents.view(encoded_all_agents.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

# === AGENT MODULE ===
class AgentModule(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            num_agents,
            hidden_dim=128,
            action_type="discrete",
            n_heads=4,
            n_layers=2,
            use_centralized_critic=False
    ):
        super().__init__()
        self.encoder = SharedAttentionEncoder(obs_dim, hidden_dim, n_heads, n_layers)
        self.policy_head = PolicyHead(hidden_dim, action_dim, action_type)
        self.value_head = ValueHead(hidden_dim)

        self.use_centralized_critic = use_centralized_critic
        if use_centralized_critic:
            self.centralized_critic = CentralizedCritic(num_agents, hidden_dim)
    
    def forward(self, obs_batch):
        encoded = self.encoder(obs_batch)
        policy_output = self.policy_head(encoded)
        value_output = self.value_head(encoded)

        outputs = {
            "policy": policy_output,
            "value": value_output
        }

        if self.use_centralized_critic:
            outputs["central_value"] = self.centralized_critic(encoded)
        
        return outputs