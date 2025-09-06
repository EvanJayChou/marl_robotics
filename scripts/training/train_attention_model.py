"""
Training script for Attention-based Multi-Agent Bipedal Locomotion

This script trains attention-based agents for bipedal locomotion tasks,
supporting both single-agent and multi-agent scenarios with communication.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import gym
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents import AttentionAgent, PPOAgent, SACAgent, MultiAgentPPO
from src.agents.communication_layer import CommunicationProtocol
from src.algorithms.curriculum_learning import CurriculumLearning
from src.algorithms.domain_adaptation import DomainAdaptation
from src.algorithms.meta_learning import MetaLearning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

