"""
Training script for Curriculum Learning in Bipedal Locomotion

This script implements curriculum learning for bipedal locomotion tasks,
progressively increasing difficulty to improve learning efficiency.
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

from src.agents import PPOAgent, SACAgent, MultiAgentPPO
from src.algorithms.curriculum_learning import CurriculumLearning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurriculumEnvironment:
    """
    Environment wrapper that supports curriculum learning by modifying
    environment parameters based on difficulty level.
    """
    
    def __init__(
        self,
        env_name: str = "BipedalWalker-v3",
        base_config: Dict[str, Any] = None,
        **kwargs
    ):
        self.env_name = env_name
        self.base_config = base_config or {}
        self.current_difficulty = 0
        self.max_difficulty = kwargs.get('max_difficulty', 5)
        
        # Create base environment
        self.env = gym.make(env_name, **self.base_config)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Curriculum parameters
        self.difficulty_params = self._create_difficulty_params()
        
        logger.info(f"Initialized CurriculumEnvironment with {self.max_difficulty} difficulty levels")
    
    def _create_difficulty_params(self) -> List[Dict[str, Any]]:
        """Create parameters for different difficulty levels."""
        # Example difficulty progression for BipedalWalker
        difficulty_params = [
            # Level 0: Easy - flat terrain, no obstacles
            {
                'terrain_type': 'flat',
                'obstacle_density': 0.0,
                'gravity': 9.8,
                'friction': 1.0,
                'reward_scale': 1.0
            },
            # Level 1: Slight hills
            {
                'terrain_type': 'hills',
                'obstacle_density': 0.0,
                'gravity': 9.8,
                'friction': 0.9,
                'reward_scale': 1.0
            },
            # Level 2: Moderate terrain with some obstacles
            {
                'terrain_type': 'moderate',
                'obstacle_density': 0.1,
                'gravity': 9.8,
                'friction': 0.8,
                'reward_scale': 1.0
            },
            # Level 3: Challenging terrain
            {
                'terrain_type': 'challenging',
                'obstacle_density': 0.2,
                'gravity': 10.0,
                'friction': 0.7,
                'reward_scale': 1.0
            },
            # Level 4: Difficult terrain with many obstacles
            {
                'terrain_type': 'difficult',
                'obstacle_density': 0.3,
                'gravity': 10.5,
                'friction': 0.6,
                'reward_scale': 1.0
            },
            # Level 5: Expert level - very challenging
            {
                'terrain_type': 'expert',
                'obstacle_density': 0.4,
                'gravity': 11.0,
                'friction': 0.5,
                'reward_scale': 1.0
            }
        ]
        
        return difficulty_params
    
    def set_difficulty(self, level: int):
        """Set the current difficulty level."""
        if 0 <= level < len(self.difficulty_params):
            self.current_difficulty = level
            # Note: In a real implementation, you would modify the environment
            # parameters here. For BipedalWalker, this might involve:
            # - Changing terrain generation
            # - Modifying physics parameters
            # - Adjusting reward functions
            logger.info(f"Set difficulty level to {level}")
        else:
            logger.warning(f"Invalid difficulty level: {level}")
    
    def get_difficulty_info(self) -> Dict[str, Any]:
        """Get information about current difficulty level."""
        if self.current_difficulty < len(self.difficulty_params):
            return self.difficulty_params[self.current_difficulty].copy()
        return {}
    
    def reset(self):
        """Reset environment with current difficulty settings."""
        obs, _ = self.env.reset()
        return obs
    
    def step(self, action):
        """Take step in environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply difficulty-based reward scaling
        difficulty_info = self.get_difficulty_info()
        reward_scale = difficulty_info.get('reward_scale', 1.0)
        reward *= reward_scale
        
        # Add difficulty information to info
        info['difficulty_level'] = self.current_difficulty
        info['difficulty_info'] = difficulty_info
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close environment."""
        self.env.close()


class CurriculumTrainer:
    """
    Trainer that implements curriculum learning for bipedal locomotion.
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
        
        # Curriculum parameters
        self.curriculum_config = config.get('curriculum_config', {})
        self.episodes_per_level = self.curriculum_config.get('episodes_per_level', 200)
        self.success_threshold = self.curriculum_config.get('success_threshold', 0.8)
        self.min_episodes_at_level = self.curriculum_config.get('min_episodes_at_level', 50)
        
        # Environment setup
        self.env_name = config.get('env_name', 'BipedalWalker-v3')
        self.env = CurriculumEnvironment(
            env_name=self.env_name,
            max_difficulty=self.curriculum_config.get('difficulty_levels', 5),
            **config.get('env_config', {})
        )
        
        # Initialize curriculum learning
        self.curriculum = CurriculumLearning(
            difficulty_levels=self.curriculum_config.get('difficulty_levels', 5),
            success_threshold=self.success_threshold,
            episodes_per_level=self.episodes_per_level,
            min_episodes_at_level=self.min_episodes_at_level
        )
        
        # Initialize agent
        algorithm = config.get('algorithm', 'ppo')
        if algorithm == 'ppo':
            self.agent = PPOAgent(
                agent_id="curriculum_ppo_agent",
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                action_type="continuous",
                device=self.device,
                **config.get('agent_config', {})
            )
        elif algorithm == 'sac':
            self.agent = SACAgent(
                agent_id="curriculum_sac_agent",
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                action_type="continuous",
                device=self.device,
                **config.get('agent_config', {})
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.difficulty_levels = []
        self.training_metrics = []
        self.curriculum_progress = []
        
        # Setup logging and checkpointing
        self._setup_logging()
        
        logger.info(f"Initialized CurriculumTrainer with {algorithm} agent")
        logger.info(f"Curriculum levels: {self.curriculum.difficulty_levels}")
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"curriculum_training_{timestamp}"
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
        """Main training loop with curriculum learning."""
        logger.info("Starting curriculum learning training...")
        
        for episode in range(self.num_episodes):
            # Update curriculum and set environment difficulty
            current_level = self.curriculum.get_current_level()
            self.env.set_difficulty(current_level)
            
            episode_reward, episode_length, metrics = self._train_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.difficulty_levels.append(current_level)
            if metrics:
                self.training_metrics.append(metrics)
            
            # Update curriculum based on performance
            success = self._evaluate_episode_success(episode_reward, episode_length)
            curriculum_updated = self.curriculum.update(episode, episode_reward, success)
            
            if curriculum_updated:
                new_level = self.curriculum.get_current_level()
                logger.info(f"Curriculum updated! New difficulty level: {new_level}")
                self.curriculum_progress.append({
                    'episode': episode,
                    'old_level': current_level,
                    'new_level': new_level,
                    'reward': episode_reward
                })
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_episode(episode, episode_reward, episode_length, metrics, current_level)
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                self._evaluate(episode)
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        # Final evaluation and save
        self._evaluate(self.num_episodes - 1)
        self._save_checkpoint(self.num_episodes - 1, final=True)
        
        logger.info("Curriculum learning training completed!")
        self._plot_training_curves()
        self._plot_curriculum_progress()
    
    def _train_episode(self, episode: int) -> Tuple[float, int, Dict[str, float]]:
        """Train for one episode."""
        obs = self.env.reset()
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
            if hasattr(self.agent, 'store_experience'):
                if 'value' in action_info and 'log_prob' in action_info:
                    # PPO agent
                    self.agent.store_experience(
                        obs=obs,
                        action=action,
                        reward=reward,
                        value=action_info['value'],
                        log_prob=action_info['log_prob'],
                        done=done
                    )
                else:
                    # SAC agent
                    self.agent.store_experience(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        done=done
                    )
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Update agent
        if hasattr(self.agent, 'update'):
            update_metrics = self.agent.update({})
            if update_metrics:
                metrics.update(update_metrics)
        
        return episode_reward, episode_length, metrics
    
    def _evaluate_episode_success(self, reward: float, length: int) -> bool:
        """Evaluate if an episode was successful based on curriculum criteria."""
        # Define success criteria based on reward and episode length
        min_reward = 200  # Minimum reward for success
        min_length = 100  # Minimum episode length for success
        
        return reward >= min_reward and length >= min_length
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float], difficulty_level: int):
        """Log episode information."""
        # Console logging
        logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}, Difficulty={difficulty_level}")
        if metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(f"  Metrics: {metric_str}")
        
        # TensorBoard logging
        self.writer.add_scalar('Episode/Reward', reward, episode)
        self.writer.add_scalar('Episode/Length', length, episode)
        self.writer.add_scalar('Curriculum/DifficultyLevel', difficulty_level, episode)
        
        if metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Training/{key}', value, episode)
        
        # Moving averages
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/AvgReward_100', avg_reward, episode)
            self.writer.add_scalar('Episode/AvgLength_100', avg_length, episode)
        
        # Curriculum statistics
        self.writer.add_scalar('Curriculum/CurrentLevel', difficulty_level, episode)
        self.writer.add_scalar('Curriculum/Progress', difficulty_level / self.curriculum.difficulty_levels, episode)
    
    def _evaluate(self, episode: int, num_eval_episodes: int = 5):
        """Evaluate the agent at current difficulty level."""
        logger.info(f"Evaluating at episode {episode}...")
        
        current_level = self.curriculum.get_current_level()
        eval_rewards = []
        eval_lengths = []
        
        for eval_episode in range(num_eval_episodes):
            obs = self.env.reset()
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
        
        logger.info(f"Evaluation (Level {current_level}): Avg Reward={avg_eval_reward:.2f}, Avg Length={avg_eval_length:.2f}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Evaluation/AvgReward', avg_eval_reward, episode)
        self.writer.add_scalar('Evaluation/AvgLength', avg_eval_length, episode)
        self.writer.add_scalar('Evaluation/DifficultyLevel', current_level, episode)
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pth")
        
        self.agent.save_checkpoint(checkpoint_path)
        
        # Save curriculum state
        curriculum_path = checkpoint_path.replace('.pth', '_curriculum.json')
        curriculum_state = {
            'current_level': self.curriculum.get_current_level(),
            'episode': episode,
            'progress': self.curriculum_progress
        }
        with open(curriculum_path, 'w') as f:
            json.dump(curriculum_state, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Saved curriculum state: {curriculum_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        if not self.episode_rewards:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward curve with difficulty levels
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_rewards)), moving_avg, color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Curriculum Learning - Training Rewards')
        ax1.grid(True)
        
        # Difficulty level progression
        ax2.plot(self.difficulty_levels, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Difficulty Level')
        ax2.set_title('Curriculum Learning - Difficulty Progression')
        ax2.set_ylim(0, max(self.difficulty_levels) + 1)
        ax2.grid(True)
        
        # Episode length curve
        ax3.plot(self.episode_lengths, alpha=0.3, color='orange')
        if len(self.episode_lengths) >= 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            ax3.plot(range(99, len(self.episode_lengths)), moving_avg, color='red', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Curriculum Learning - Episode Lengths')
        ax3.grid(True)
        
        # Training metrics
        if self.training_metrics:
            policy_losses = [m.get('policy_loss', 0) for m in self.training_metrics]
            value_losses = [m.get('value_loss', 0) for m in self.training_metrics]
            
            ax4.plot(policy_losses, label='Policy Loss', alpha=0.7)
            ax4.plot(value_losses, label='Value Loss', alpha=0.7)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.set_title('Curriculum Learning - Training Losses')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves: {plot_path}")
    
    def _plot_curriculum_progress(self):
        """Plot curriculum progression."""
        if not self.curriculum_progress:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        episodes = [p['episode'] for p in self.curriculum_progress]
        levels = [p['new_level'] for p in self.curriculum_progress]
        rewards = [p['reward'] for p in self.curriculum_progress]
        
        # Plot curriculum updates
        ax.scatter(episodes, levels, c=rewards, cmap='viridis', s=100, alpha=0.7)
        ax.plot(episodes, levels, 'r--', alpha=0.5)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Difficulty Level')
        ax.set_title('Curriculum Learning - Level Progression')
        ax.set_ylim(0, max(levels) + 1)
        ax.grid(True)
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Episode Reward')
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "curriculum_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved curriculum progress: {plot_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Environment
        'env_name': 'BipedalWalker-v3',
        'env_config': {},
        
        # Training
        'num_episodes': 1000,
        'max_steps_per_episode': 1000,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 10,
        
        # Algorithm
        'algorithm': 'ppo',  # 'ppo' or 'sac'
        
        # Curriculum learning
        'curriculum_config': {
            'difficulty_levels': 5,
            'success_threshold': 0.8,
            'episodes_per_level': 200,
            'min_episodes_at_level': 50
        },
        
        # Agent configuration
        'agent_config': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'hidden_dim': 128,
            'num_layers': 2,
            'batch_size': 64,
            'ppo_epochs': 4,
            'clip_ratio': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'gae_lambda': 0.95,
            'action_bounds': (-1.0, 1.0)
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train agent with curriculum learning for bipedal locomotion')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment name')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'], help='Training algorithm')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--difficulty-levels', type=int, default=5, help='Number of difficulty levels')
    parser.add_argument('--episodes-per-level', type=int, default=200, help='Episodes per difficulty level')
    parser.add_argument('--success-threshold', type=float, default=0.8, help='Success threshold for curriculum')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config['env_name'] = args.env
    config['algorithm'] = args.algorithm
    config['num_episodes'] = args.episodes
    config['curriculum_config']['difficulty_levels'] = args.difficulty_levels
    config['curriculum_config']['episodes_per_level'] = args.episodes_per_level
    config['curriculum_config']['success_threshold'] = args.success_threshold
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
    trainer = CurriculumTrainer(config, device)
    trainer.train()
    
    # Cleanup
    trainer.env.close()
    trainer.writer.close()


if __name__ == "__main__":
    main()
