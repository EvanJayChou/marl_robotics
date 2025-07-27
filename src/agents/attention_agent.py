import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neural_networks.attention_models import (
    SharedAttentionEncoder, 
    PolicyHead, 
    ValueHead, 
    CentralizedCritic
)

# === ATTENTION-BASED AGENT MODULE ===
class AttentionAgent(nn.Module):
    """
    Attention-based agent that combines shared attention encoding with policy and value heads.
    This agent can optionally use a centralized critic for multi-agent scenarios.
    """
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
        """
        Forward pass through the attention agent.
        
        Args:
            obs_batch: Observation batch of shape (batch_size, seq_len, obs_dim)
            
        Returns:
            dict: Contains policy outputs, value outputs, and optionally centralized critic outputs
        """
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
    
    def get_action(self, obs, deterministic=False):
        """
        Get action from the agent.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic action selection
            
        Returns:
            tuple: (action, action_log_prob, value)
        """
        outputs = self.forward(obs)
        
        if self.policy_head.action_type == "discrete":
            action_probs = F.softmax(outputs["policy"], dim=-1)
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            action_log_prob = torch.log(action_probs + 1e-8).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            action_mean, action_std = outputs["policy"]
            if deterministic:
                action = action_mean
            else:
                action_dist = torch.distributions.Normal(action_mean, action_std)
                action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        return action, action_log_prob, outputs["value"] 