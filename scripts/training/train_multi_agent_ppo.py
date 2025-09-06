"""
Training script for Multi-Agent PPO in Bipedal Locomotion

This script trains multiple PPO agents with communication and coordination
for bipedal locomotion tasks.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
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

from src.agents import MultiAgentPPO
from src.agents.communication_layer import CommunicationProtocol
from src.algorithms.curriculum_learning import CurriculumLearning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentEnvironment:
    """
    Multi-agent environment wrapper for bipedal locomotion.
    """
    
    def __init__(
        self,
        env_name: str = "BipedalWalker-v3",
        num_agents: int = 2,
        render: bool = False,
        **kwargs
    ):
        self.env_name = env_name
        self.num_agents = num_agents
        self.render = render
        
        # Initialize multiple environments
        self.envs = [gym.make(env_name) for _ in range(num_agents)]
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.action_dim = self.envs[0].action_space.shape[0]
        
        logger.info(f"Initialized {num_agents} agent environment with obs_dim={self.obs_dim}, action_dim={self.action_dim}")
    
    def reset(self) -> np.ndarray:
        """Reset all environments and return initial observations."""
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.array(obs_list)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, List[float], List[bool], List[Dict]]:
        """Take step in all environments."""
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.array(obs_list), rewards, dones, infos
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class MultiAgentPPOTrainer:
    """
    Trainer for multi-agent PPO system in bipedal locomotion.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        
        # Training parameters
        self.num_episodes = config.get('num_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.save_frequency = config.get('save_frequency', 100)
        self.eval_frequency = config.get('eval_frequency', 50)
        self.log_frequency = config.get('log_frequency', 10)
        
        # Multi-agent parameters
        self.num_agents = config.get('num_agents', 2)
        self.communication_type = config.get('communication_type', 'attention')
        self.coordination_type = config.get('coordination_type', 'centralized')
        
        # Environment setup
        self.env_name = config.get('env_name', 'BipedalWalker-v3')
        self.env = MultiAgentEnvironment(
            env_name=self.env_name,
            num_agents=self.num_agents,
            render=config.get('render', False)
        )
        
        # Initialize multi-agent PPO system
        agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.multi_agent = MultiAgentPPO(
            agent_ids=agent_ids,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            action_type="continuous",
            device=self.device,
            num_agents=self.num_agents,
            communication_type=self.communication_type,
            coordination_type=self.coordination_type,
            **config.get('agent_config', {})
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.agent_rewards = [[] for _ in range(self.num_agents)]
        self.training_metrics = []
        self.communication_metrics = []
        
        # Setup logging and checkpointing
        self._setup_logging()
        
        # Curriculum learning (optional)
        self.curriculum = None
        if config.get('use_curriculum', False):
            self.curriculum = CurriculumLearning(**config.get('curriculum_config', {}))
        
        logger.info(f"Initialized MultiAgentPPOTrainer with {self.num_agents} agents")
        logger.info(f"Communication: {self.communication_type}, Coordination: {self.coordination_type}")
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"multi_agent_ppo_{self.num_agents}agents_{timestamp}"
        self.log_dir = os.path.join("logs", self.run_name)
        self.checkpoint_dir = os.path.join("checkpoints", self.run_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Save configuration
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Logging to {self.log_dir}")
        logger.info(f"Checkpoints saved to {self.checkpoint_dir}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Multi-Agent PPO training...")
        
        for episode in range(self.num_episodes):
            episode_reward, episode_length, metrics = self._train_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if metrics:
                self.training_metrics.append(metrics)
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_episode(episode, episode_reward, episode_length, metrics)
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                self._evaluate(episode)
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Curriculum learning update
            if self.curriculum:
                self.curriculum.update(episode, episode_reward)
        
        # Final evaluation and save
        self._evaluate(self.num_episodes - 1)
        self._save_checkpoint(self.num_episodes - 1, final=True)
        
        logger.info("Multi-Agent PPO training completed!")
        self._plot_training_curves()
    
    def _train_episode(self, episode: int) -> Tuple[float, int, Dict[str, float]]:
        """Train for one episode."""
        obs_all = self.env.reset()
        episode_reward = 0
        episode_length = 0
        metrics = {}
        agent_rewards = [0.0] * self.num_agents
        
        for step in range(self.max_steps_per_episode):
            # Select actions for all agents
            actions, action_info = self.multi_agent.select_action(obs_all)
            
            # Environment step
            next_obs_all, rewards, dones, infos = self.env.step(actions)
            
            # Store experience for each agent
            for i, agent_id in enumerate(self.multi_agent.agent_ids):
                self.multi_agent.store_experience(
                    agent_id=agent_id,
                    obs=obs_all[i],
                    action=actions[i],
                    reward=rewards[i],
                    value=action_info[agent_id]['value'],
                    log_prob=action_info[agent_id]['log_prob'],
                    done=dones[i]
                )
                
                agent_rewards[i] += rewards[i]
            
            episode_reward += sum(rewards)
            episode_length += 1
            obs_all = next_obs_all
            
            # Check if episode is done
            if all(dones):
                break
        
        # Store agent rewards
        for i in range(self.num_agents):
            self.agent_rewards[i].append(agent_rewards[i])
        
        # Update multi-agent system
        update_metrics = self.multi_agent.update({})
        if update_metrics:
            metrics.update(update_metrics)
        
        # Communication metrics
        if hasattr(self.multi_agent, 'communication_protocol') and self.multi_agent.communication_protocol:
            comm_stats = self.multi_agent.communication_protocol.get_communication_stats()
            if comm_stats:
                metrics.update({f"comm_{k}": v for k, v in comm_stats.items()})
        
        return episode_reward, episode_length, metrics
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float]):
        """Log episode information."""
        # Console logging
        logger.info(f"Episode {episode}: Total Reward={reward:.2f}, Length={length}")
        
        # Individual agent rewards
        for i in range(self.num_agents):
            agent_reward = self.agent_rewards[i][-1] if self.agent_rewards[i] else 0
            logger.info(f"  Agent {i}: Reward={agent_reward:.2f}")
        
        if metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(f"  Metrics: {metric_str}")
        
        # TensorBoard logging
        self.writer.add_scalar('Episode/TotalReward', reward, episode)
        self.writer.add_scalar('Episode/Length', length, episode)
        
        # Individual agent rewards
        for i in range(self.num_agents):
            agent_reward = self.agent_rewards[i][-1] if self.agent_rewards[i] else 0
            self.writer.add_scalar(f'Episode/Agent_{i}_Reward', agent_reward, episode)
        
        if metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Training/{key}', value, episode)
        
        # Moving averages
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/AvgTotalReward_100', avg_reward, episode)
            self.writer.add_scalar('Episode/AvgLength_100', avg_length, episode)
            
            # Individual agent moving averages
            for i in range(self.num_agents):
                if len(self.agent_rewards[i]) >= 100:
                    avg_agent_reward = np.mean(self.agent_rewards[i][-100:])
                    self.writer.add_scalar(f'Episode/AvgAgent_{i}_Reward_100', avg_agent_reward, episode)
    
    def _evaluate(self, episode: int, num_eval_episodes: int = 5):
        """Evaluate the multi-agent system."""
        logger.info(f"Evaluating at episode {episode}...")
        
        eval_rewards = []
        eval_lengths = []
        eval_agent_rewards = [[] for _ in range(self.num_agents)]
        
        for eval_episode in range(num_eval_episodes):
            obs_all = self.env.reset()
            episode_reward = 0
            episode_length = 0
            agent_rewards = [0.0] * self.num_agents
            
            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    actions, _ = self.multi_agent.select_action(obs_all, deterministic=True)
                
                next_obs_all, rewards, dones, _ = self.env.step(actions)
                
                episode_reward += sum(rewards)
                episode_length += 1
                
                for i in range(self.num_agents):
                    agent_rewards[i] += rewards[i]
                
                obs_all = next_obs_all
                
                if all(dones):
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            for i in range(self.num_agents):
                eval_agent_rewards[i].append(agent_rewards[i])
        
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_length = np.mean(eval_lengths)
        avg_agent_rewards = [np.mean(eval_agent_rewards[i]) for i in range(self.num_agents)]
        
        logger.info(f"Evaluation: Avg Total Reward={avg_eval_reward:.2f}, Avg Length={avg_eval_length:.2f}")
        for i in range(self.num_agents):
            logger.info(f"  Agent {i}: Avg Reward={avg_agent_rewards[i]:.2f}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Evaluation/AvgTotalReward', avg_eval_reward, episode)
        self.writer.add_scalar('Evaluation/AvgLength', avg_eval_length, episode)
        
        for i in range(self.num_agents):
            self.writer.add_scalar(f'Evaluation/AvgAgent_{i}_Reward', avg_agent_rewards[i], episode)
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pth")
        
        self.multi_agent.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        if not self.episode_rewards:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total reward curve
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_rewards)), moving_avg, color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Multi-Agent PPO Training - Total Rewards')
        ax1.grid(True)
        
        # Individual agent rewards
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i in range(self.num_agents):
            if self.agent_rewards[i]:
                ax2.plot(self.agent_rewards[i], alpha=0.3, color=colors[i % len(colors)], label=f'Agent {i}')
                if len(self.agent_rewards[i]) >= 100:
                    moving_avg = np.convolve(self.agent_rewards[i], np.ones(100)/100, mode='valid')
                    ax2.plot(range(99, len(self.agent_rewards[i])), moving_avg, color=colors[i % len(colors)], linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Agent Reward')
        ax2.set_title('Multi-Agent PPO Training - Individual Agent Rewards')
        ax2.legend()
        ax2.grid(True)
        
        # Episode length curve
        ax3.plot(self.episode_lengths, alpha=0.3, color='green')
        if len(self.episode_lengths) >= 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            ax3.plot(range(99, len(self.episode_lengths)), moving_avg, color='red', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Multi-Agent PPO Training - Episode Lengths')
        ax3.grid(True)
        
        # Training metrics
        if self.training_metrics:
            policy_losses = [m.get('policy_loss', 0) for m in self.training_metrics]
            value_losses = [m.get('value_loss', 0) for m in self.training_metrics]
            
            ax4.plot(policy_losses, label='Policy Loss', alpha=0.7)
            ax4.plot(value_losses, label='Value Loss', alpha=0.7)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.set_title('Multi-Agent PPO Training - Losses')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves: {plot_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Environment
        'env_name': 'BipedalWalker-v3',
        'num_agents': 2,
        'render': False,
        
        # Training
        'num_episodes': 1000,
        'max_steps_per_episode': 1000,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 10,
        
        # Multi-agent parameters
        'communication_type': 'attention',
        'coordination_type': 'centralized',
        
        # Curriculum learning
        'use_curriculum': False,
        'curriculum_config': {
            'difficulty_levels': 5,
            'success_threshold': 0.8,
            'episodes_per_level': 200
        },
        
        # Multi-Agent PPO configuration
        'agent_config': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,
            'batch_size': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'action_bounds': (-1.0, 1.0),
            'message_dim': 64,
            'num_attention_heads': 4,
            'use_centralized_critic': True,
            'use_shared_policy': False
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train multi-agent PPO for bipedal locomotion')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment name')
    parser.add_argument('--num-agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--communication-type', type=str, default='attention', 
                       choices=['attention', 'graph', 'none'], help='Communication type')
    parser.add_argument('--coordination-type', type=str, default='centralized',
                       choices=['centralized', 'decentralized', 'hierarchical'], help='Coordination type')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--use-curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--centralized-critic', action='store_true', default=True, help='Use centralized critic')
    parser.add_argument('--shared-policy', action='store_true', help='Use shared policy across agents')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config['env_name'] = args.env
    config['num_agents'] = args.num_agents
    config['num_episodes'] = args.episodes
    config['communication_type'] = args.communication_type
    config['coordination_type'] = args.coordination_type
    config['use_curriculum'] = args.use_curriculum
    config['agent_config']['learning_rate'] = args.learning_rate
    config['agent_config']['batch_size'] = args.batch_size
    config['agent_config']['hidden_dim'] = args.hidden_dim
    config['agent_config']['use_centralized_critic'] = args.centralized_critic
    config['agent_config']['use_shared_policy'] = args.shared_policy
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Create trainer and start training
    trainer = MultiAgentPPOTrainer(config, device)
    trainer.train()
    
    # Cleanup
    trainer.env.close()
    trainer.writer.close()


if __name__ == "__main__":
    main()
