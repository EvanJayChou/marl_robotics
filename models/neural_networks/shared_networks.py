import torch
import torch.nn as nn
import torch.nn.functional as F

# === SHARED ENCODER ===
# Used to encode observations from multiple agents using the same weights
class SharedEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, obs):
        return self.net(obs)

# === SHARED VALUE NETWORK ===
# Used by a entralized critic in MAPPO to evaluate global state info
class SharedValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.value_net(x)

# === SHARED ATTENTION ===
# Used in attention-based multiagent RL
class SharedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        attn_output, _  = self.attn(x, x, x)
        return attn_output

# === SHARED RNN ===
# Used for temporal sequence modeling
class SharedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='gru'):
        super().__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
    
    def forward(self, x, h=None):
        output, h_out = self.rnn(x, h)
        return output, h_out