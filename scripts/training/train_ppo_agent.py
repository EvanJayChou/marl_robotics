"""
Training script for PPO Agent in Bipedal Locomotion

This script trains a single PPO agent for bipedal locomotion tasks
with comprehensive logging, evaluation, and checkpointing.
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

from src.agents import PPOAgent
from src.algorithms.curriculum_learning import CurriculumLearning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Trainer for PPO agent in bipedal locomotion.
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
        
        # Environment setup
        self.env_name = config.get('env_name', 'BipedalWalker-v3')
        self.env = gym.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            agent_id="ppo_bipedal_agent",
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_type="continuous",
            device=self.device,
            **config.get('agent_config', {})
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
        # Setup logging and checkpointing
        self._setup_logging()
        
        # Curriculum learning (optional)
        self.curriculum = None
        if config.get('use_curriculum', False):
            self.curriculum = CurriculumLearning(**config.get('curriculum_config', {}))
        
        logger.info(f"Initialized PPOTrainer for {self.env_name}")
        logger.info(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"ppo_training_{timestamp}"
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
        logger.info("Starting PPO training...")
        
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
        
        logger.info("PPO training completed!")
        self._plot_training_curves()
    
    def _train_episode(self, episode: int) -> Tuple[float, int, Dict[str, float]]:
        """Train for one episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        metrics = {}
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action, action_info = self.agent.select_action(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.agent.store_experience(
                obs=obs,
                action=action,
                reward=reward,
                value=action_info['value'],
                log_prob=action_info['log_prob'],
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Update agent
        update_metrics = self.agent.update({})
        if update_metrics:
            metrics.update(update_metrics)
        
        return episode_reward, episode_length, metrics
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float]):
        """Log episode information."""
        # Console logging
        logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}")
        if metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(f"  Metrics: {metric_str}")
        
        # TensorBoard logging
        self.writer.add_scalar('Episode/Reward', reward, episode)
        self.writer.add_scalar('Episode/Length', length, episode)
        
        if metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Training/{key}', value, episode)
        
        # Moving averages
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/AvgReward_100', avg_reward, episode)
            self.writer.add_scalar('Episode/AvgLength_100', avg_length, episode)
    
    def _evaluate(self, episode: int, num_eval_episodes: int = 5):
        """Evaluate the agent."""
        logger.info(f"Evaluating at episode {episode}...")
        
        eval_rewards = []
        eval_lengths = []
        
        for eval_episode in range(num_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    action, _ = self.agent.select_action(obs, deterministic=True)
                
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_length = np.mean(eval_lengths)
        
        logger.info(f"Evaluation: Avg Reward={avg_eval_reward:.2f}, Avg Length={avg_eval_length:.2f}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Evaluation/AvgReward', avg_eval_reward, episode)
        self.writer.add_scalar('Evaluation/AvgLength', avg_eval_length, episode)
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pth")
        
        self.agent.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        if not self.episode_rewards:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Reward curve
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_rewards)), moving_avg, color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('PPO Training Rewards')
        ax1.grid(True)
        
        # Length curve
        ax2.plot(self.episode_lengths, alpha=0.3, color='green')
        if len(self.episode_lengths) >= 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            ax2.plot(range(99, len(self.episode_lengths)), moving_avg, color='red', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('PPO Training Episode Lengths')
        ax2.grid(True)
        
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
        
        # Training
        'num_episodes': 1000,
        'max_steps_per_episode': 1000,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 10,
        
        # Curriculum learning
        'use_curriculum': False,
        'curriculum_config': {
            'difficulty_levels': 5,
            'success_threshold': 0.8,
            'episodes_per_level': 200
        },
        
        # PPO Agent configuration
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
            'action_bounds': (-1.0, 1.0)
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train PPO agent for bipedal locomotion')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--use-curriculum', action='store_true', help='Use curriculum learning')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config['env_name'] = args.env
    config['num_episodes'] = args.episodes
    config['use_curriculum'] = args.use_curriculum
    config['agent_config']['learning_rate'] = args.learning_rate
    config['agent_config']['batch_size'] = args.batch_size
    config['agent_config']['hidden_dim'] = args.hidden_dim
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Create trainer and start training
    trainer = PPOTrainer(config, device)
    trainer.train()
    
    # Cleanup
    trainer.env.close()
    trainer.writer.close()


if __name__ == "__main__":
    main()
