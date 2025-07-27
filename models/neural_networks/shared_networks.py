import torch
import torch.nn as nn
import torch.nn.functional as F

# === SHARED ENCODER ===
class SharedEncoder(nn.Module):
    """
    Shared encoder used to encode observations from multiple agents using the same weights.
    This enables parameter sharing across agents for more efficient learning.
    """
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
class SharedValueNetwork(nn.Module):
    """
    Shared value network used by centralized critics in MAPPO to evaluate global state information.
    """
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
class SharedAttention(nn.Module):
    """
    Shared attention mechanism used in attention-based multi-agent RL.
    Enables agents to attend to each other's observations or states.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

# === SHARED RNN ===
class SharedRNN(nn.Module):
    """
    Shared RNN used for temporal sequence modeling in multi-agent scenarios.
    Supports both GRU and LSTM architectures.
    """
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

# === SHARED TRANSFORMER ===
class SharedTransformer(nn.Module):
    """
    Shared transformer encoder for processing sequences of agent observations.
    """
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=6):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.input_projection(x)
        return self.transformer(x)

# === SHARED CONVOLUTIONAL ENCODER ===
class SharedConvEncoder(nn.Module):
    """
    Shared convolutional encoder for processing spatial observations.
    """
    def __init__(self, input_channels, hidden_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, hidden_dim)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)