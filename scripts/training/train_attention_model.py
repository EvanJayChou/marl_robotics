from src.agents.attention_agent import AttentionAgent
import torch
import torch.nn as nn
import torch.nn.functional as F



def train():
    for epoch in range(num_epochs):
        