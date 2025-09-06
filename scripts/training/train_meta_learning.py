"""
Training script for Meta-Learning in Bipedal Locomotion

This script implements meta-learning algorithms (MAML, Reptile) for
fast adaptation to new bipedal locomotion tasks.
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

from src.agents import PPOAgent, SACAgent
from src.algorithms.meta_learning import MetaLearning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetaLearningEnvironment:
    """
    Environment wrapper that supports meta-learning with multiple tasks.
    """
    
    def __init__(
        self,
        base_env_name: str = "BipedalWalker-v3",
        num_tasks: int = 5,
        task_config: Dict[str, Any] = None,
        **kwargs
    ):
        self.base_env_name = base_env_name
        self.num_tasks = num_tasks
        self.task_config = task_config or {}
        
        # Create base environment
        self.base_env = gym.make(base_env_name)
        self.obs_dim = self.base_env.observation_space.shape[0]
        self.action_dim = self.base_env.action_space.shape[0]
        
        # Task parameters
        self.tasks = self._create_tasks()
        self.current_task = 0
        
        logger.info(f"Initialized MetaLearningEnvironment with {num_tasks} tasks")
    
    def _create_tasks(self) -> List[Dict[str, Any]]:
        """Create different tasks for meta-learning."""
        tasks = []
        
        for i in range(self.num_tasks):
            task = {
                'task_id': i,
                'name': f'task_{i}',
                'gravity': 9.8 + np.random.uniform(-1.0, 1.0),
                'friction': 0.8 + np.random.uniform(-0.2, 0.2),
                'reward_scale': 1.0 + np.random.uniform(-0.2, 0.2),
                'terrain_type': np.random.choice(['flat', 'hills', 'rough']),
                'obstacle_density': np.random.uniform(0.0, 0.3),
                'target_velocity': np.random.uniform(0.5, 2.0),
                'energy_penalty': np.random.uniform(0.0, 0.1)
            }
            tasks.append(task)
        
        return tasks
    
    def set_task(self, task_id: int):
        """Set the current task."""
        if 0 <= task_id < self.num_tasks:
            self.current_task = task_id
            logger.info(f"Switched to task {task_id}: {self.tasks[task_id]['name']}")
        else:
            logger.warning(f"Invalid task ID: {task_id}")
    
    def get_current_task(self) -> Dict[str, Any]:
        """Get current task parameters."""
        return self.tasks[self.current_task].copy()
    
    def reset(self):
        """Reset environment with current task parameters."""
        obs, _ = self.base_env.reset()
        
        # Apply task-specific modifications
        task_params = self.get_current_task()
        obs = self._apply_task_modifications(obs, task_params)
        
        return obs
    
    def step(self, action):
        """Take step in environment with current task parameters."""
        task_params = self.get_current_task()
        
        # Apply task-specific action modifications
        action = self._apply_action_modifications(action, task_params)
        
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Apply task-specific observation modifications
        obs = self._apply_task_modifications(obs, task_params)
        
        # Apply task-specific reward modifications
        reward = self._apply_reward_modifications(reward, task_params, info)
        
        # Add task information to info
        info['task_id'] = self.current_task
        info['task_params'] = task_params
        
        return obs, reward, terminated, truncated, info
    
    def _apply_task_modifications(self, obs: np.ndarray, task_params: Dict[str, Any]) -> np.ndarray:
        """Apply task-specific modifications to observations."""
        # In a real implementation, this would modify observations based on task parameters
        # For now, we'll just add some noise based on terrain type
        terrain_type = task_params.get('terrain_type', 'flat')
        if terrain_type == 'rough':
            noise = np.random.normal(0, 0.01, obs.shape)
            obs = obs + noise
        
        return obs
    
    def _apply_action_modifications(self, action: np.ndarray, task_params: Dict[str, Any]) -> np.ndarray:
        """Apply task-specific modifications to actions."""
        # Apply friction-based modifications
        friction = task_params.get('friction', 1.0)
        action = action * friction
        
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _apply_reward_modifications(self, reward: float, task_params: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Apply task-specific modifications to rewards."""
        # Apply reward scaling
        reward_scale = task_params.get('reward_scale', 1.0)
        reward *= reward_scale
        
        # Apply energy penalty
        energy_penalty = task_params.get('energy_penalty', 0.0)
        if 'action' in info:
            energy_cost = np.sum(np.square(info['action']))
            reward -= energy_penalty * energy_cost
        
        return reward
    
    def close(self):
        """Close environment."""
        self.base_env.close()


class MetaLearningTrainer:
    """
    Trainer that implements meta-learning for bipedal locomotion.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        
        # Training parameters
        self.num_meta_episodes = config.get('num_meta_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.save_frequency = config.get('save_frequency', 100)
        self.eval_frequency = config.get('eval_frequency', 50)
        self.log_frequency = config.get('log_frequency', 10)
        
        # Meta-learning parameters
        self.meta_config = config.get('meta_config', {})
        self.num_tasks = self.meta_config.get('num_tasks', 5)
        self.inner_lr = self.meta_config.get('inner_lr', 1e-3)
        self.outer_lr = self.meta_config.get('outer_lr', 1e-4)
        self.num_inner_steps = self.meta_config.get('num_inner_steps', 5)
        self.adaptation_episodes = self.meta_config.get('adaptation_episodes', 10)
        
        # Environment setup
        self.env_name = config.get('env_name', 'BipedalWalker-v3')
        self.env = MetaLearningEnvironment(
            base_env_name=self.env_name,
            num_tasks=self.num_tasks,
            **config.get('env_config', {})
        )
        
        # Initialize meta-learning
        self.meta_learning = MetaLearning(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            **self.meta_config
        )
        
        # Initialize base agent
        algorithm = config.get('algorithm', 'ppo')
        if algorithm == 'ppo':
            self.base_agent = PPOAgent(
                agent_id="meta_learning_ppo_agent",
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                action_type="continuous",
                device=self.device,
                **config.get('agent_config', {})
            )
        elif algorithm == 'sac':
            self.base_agent = SACAgent(
                agent_id="meta_learning_sac_agent",
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                action_type="continuous",
                device=self.device,
                **config.get('agent_config', {})
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Training tracking
        self.meta_episode_rewards = []
        self.adaptation_rewards = []
        self.task_performances = [[] for _ in range(self.num_tasks)]
        self.meta_metrics = []
        self.adaptation_metrics = []
        
        # Setup logging and checkpointing
        self._setup_logging()
        
        logger.info(f"Initialized MetaLearningTrainer with {algorithm} agent")
        logger.info(f"Number of tasks: {self.num_tasks}")
        logger.info(f"Inner LR: {self.inner_lr}, Outer LR: {self.outer_lr}")
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"meta_learning_{timestamp}"
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
        """Main meta-learning training loop."""
        logger.info("Starting meta-learning training...")
        
        for meta_episode in range(self.num_meta_episodes):
            # Sample a batch of tasks
            task_batch = self._sample_task_batch()
            
            # Meta-learning update
            meta_reward, meta_metrics = self._meta_learning_step(task_batch, meta_episode)
            
            # Store metrics
            self.meta_episode_rewards.append(meta_reward)
            if meta_metrics:
                self.meta_metrics.append(meta_metrics)
            
            # Logging
            if meta_episode % self.log_frequency == 0:
                self._log_meta_episode(meta_episode, meta_reward, meta_metrics)
            
            # Evaluation
            if meta_episode % self.eval_frequency == 0 and meta_episode > 0:
                self._evaluate_meta_learning(meta_episode)
            
            # Save checkpoint
            if meta_episode % self.save_frequency == 0 and meta_episode > 0:
                self._save_checkpoint(meta_episode)
        
        # Final evaluation and save
        self._evaluate_meta_learning(self.num_meta_episodes - 1)
        self._save_checkpoint(self.num_meta_episodes - 1, final=True)
        
        logger.info("Meta-learning training completed!")
        self._plot_meta_learning_curves()
    
    def _sample_task_batch(self) -> List[int]:
        """Sample a batch of tasks for meta-learning."""
        # Sample tasks for meta-learning (typically 2-5 tasks)
        batch_size = min(3, self.num_tasks)
        return np.random.choice(self.num_tasks, size=batch_size, replace=False).tolist()
    
    def _meta_learning_step(self, task_batch: List[int], meta_episode: int) -> Tuple[float, Dict[str, float]]:
        """Perform one meta-learning step."""
        meta_reward = 0
        meta_metrics = {}
        
        # Store original parameters
        original_params = self._get_agent_parameters()
        
        # Inner loop: adapt to each task
        task_gradients = []
        task_rewards = []
        
        for task_id in task_batch:
            # Set environment to current task
            self.env.set_task(task_id)
            
            # Adapt agent to this task
            adapted_params, task_reward, adaptation_metrics = self._adapt_to_task(task_id)
            task_gradients.append(adapted_params)
            task_rewards.append(task_reward)
            
            # Store adaptation metrics
            if adaptation_metrics:
                self.adaptation_metrics.append(adaptation_metrics)
        
        # Outer loop: update meta-parameters
        if task_gradients:
            meta_metrics = self.meta_learning.update_meta_parameters(
                task_gradients, original_params
            )
            meta_reward = np.mean(task_rewards)
        
        return meta_reward, meta_metrics
    
    def _adapt_to_task(self, task_id: int) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, float]]:
        """Adapt agent to a specific task."""
        # Create a copy of the agent for adaptation
        adapted_agent = self._create_adapted_agent()
        
        # Collect experience and adapt
        task_rewards = []
        adaptation_metrics = {}
        
        for episode in range(self.adaptation_episodes):
            episode_reward, episode_length, metrics = self._train_episode(adapted_agent, task_id)
            task_rewards.append(episode_reward)
            
            if metrics:
                adaptation_metrics.update(metrics)
        
        # Get adapted parameters
        adapted_params = self._get_agent_parameters(adapted_agent)
        avg_task_reward = np.mean(task_rewards)
        
        # Store task performance
        self.task_performances[task_id].append(avg_task_reward)
        
        return adapted_params, avg_task_reward, adaptation_metrics
    
    def _create_adapted_agent(self):
        """Create a copy of the agent for adaptation."""
        # In a real implementation, this would create a proper copy
        # For now, we'll return the base agent
        return self.base_agent
    
    def _get_agent_parameters(self, agent=None):
        """Get agent parameters."""
        if agent is None:
            agent = self.base_agent
        
        # Get policy network parameters
        if hasattr(agent, 'policy_net'):
            return {name: param.clone() for name, param in agent.policy_net.named_parameters()}
        elif hasattr(agent, 'actor'):
            return {name: param.clone() for name, param in agent.actor.named_parameters()}
        else:
            return {}
    
    def _train_episode(self, agent, task_id: int) -> Tuple[float, int, Dict[str, float]]:
        """Train for one episode with a specific agent and task."""
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        metrics = {}
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action, action_info = agent.select_action(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            if hasattr(agent, 'store_experience'):
                if 'value' in action_info and 'log_prob' in action_info:
                    # PPO agent
                    agent.store_experience(
                        obs=obs,
                        action=action,
                        reward=reward,
                        value=action_info['value'],
                        log_prob=action_info['log_prob'],
                        done=done
                    )
                else:
                    # SAC agent
                    agent.store_experience(
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
        if hasattr(agent, 'update'):
            update_metrics = agent.update({})
            if update_metrics:
                metrics.update(update_metrics)
        
        return episode_reward, episode_length, metrics
    
    def _log_meta_episode(self, meta_episode: int, meta_reward: float, meta_metrics: Dict[str, float]):
        """Log meta-learning episode information."""
        # Console logging
        logger.info(f"Meta Episode {meta_episode}: Meta Reward={meta_reward:.2f}")
        if meta_metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in meta_metrics.items()])
            logger.info(f"  Meta Metrics: {metric_str}")
        
        # TensorBoard logging
        self.writer.add_scalar('MetaLearning/MetaReward', meta_reward, meta_episode)
        
        if meta_metrics:
            for key, value in meta_metrics.items():
                self.writer.add_scalar(f'MetaLearning/{key}', value, meta_episode)
        
        # Task performance
        for task_id in range(self.num_tasks):
            if self.task_performances[task_id]:
                avg_performance = np.mean(self.task_performances[task_id][-10:])  # Last 10 episodes
                self.writer.add_scalar(f'TaskPerformance/Task_{task_id}', avg_performance, meta_episode)
    
    def _evaluate_meta_learning(self, meta_episode: int, num_eval_episodes: int = 5):
        """Evaluate meta-learning performance."""
        logger.info(f"Evaluating meta-learning at episode {meta_episode}...")
        
        eval_rewards = []
        
        for task_id in range(self.num_tasks):
            self.env.set_task(task_id)
            task_rewards = []
            
            for eval_episode in range(num_eval_episodes):
                obs = self.env.reset()
                episode_reward = 0
                
                for step in range(self.max_steps_per_episode):
                    with torch.no_grad():
                        action, _ = self.base_agent.select_action(obs, deterministic=True)
                    
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                task_rewards.append(episode_reward)
            
            avg_task_reward = np.mean(task_rewards)
            eval_rewards.append(avg_task_reward)
            
            logger.info(f"  Task {task_id}: Avg Reward={avg_task_reward:.2f}")
        
        avg_eval_reward = np.mean(eval_rewards)
        logger.info(f"Meta-learning evaluation: Avg Reward={avg_eval_reward:.2f}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Evaluation/MetaAvgReward', avg_eval_reward, meta_episode)
        for task_id, task_reward in enumerate(eval_rewards):
            self.writer.add_scalar(f'Evaluation/Task_{task_id}_Reward', task_reward, meta_episode)
    
    def _save_checkpoint(self, meta_episode: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_meta_episode_{meta_episode}.pth")
        
        self.base_agent.save_checkpoint(checkpoint_path)
        
        # Save meta-learning state
        meta_state_path = checkpoint_path.replace('.pth', '_meta_state.json')
        meta_state = {
            'meta_episode': meta_episode,
            'task_performances': self.task_performances,
            'meta_metrics': self.meta_metrics[-10:] if self.meta_metrics else []
        }
        with open(meta_state_path, 'w') as f:
            json.dump(meta_state, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Saved meta-learning state: {meta_state_path}")
    
    def _plot_meta_learning_curves(self):
        """Plot and save meta-learning curves."""
        if not self.meta_episode_rewards:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Meta-learning reward curve
        ax1.plot(self.meta_episode_rewards, alpha=0.3, color='blue')
        if len(self.meta_episode_rewards) >= 100:
            moving_avg = np.convolve(self.meta_episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.meta_episode_rewards)), moving_avg, color='red', linewidth=2)
        ax1.set_xlabel('Meta Episode')
        ax1.set_ylabel('Meta Reward')
        ax1.set_title('Meta-Learning - Meta Rewards')
        ax1.grid(True)
        
        # Task performance curves
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for task_id in range(self.num_tasks):
            if self.task_performances[task_id]:
                ax2.plot(self.task_performances[task_id], 
                        color=colors[task_id % len(colors)], 
                        label=f'Task {task_id}', alpha=0.7)
        ax2.set_xlabel('Adaptation Episode')
        ax2.set_ylabel('Task Reward')
        ax2.set_title('Meta-Learning - Task Performance')
        ax2.legend()
        ax2.grid(True)
        
        # Meta-learning metrics
        if self.meta_metrics:
            meta_losses = [m.get('meta_loss', 0) for m in self.meta_metrics]
            adaptation_losses = [m.get('adaptation_loss', 0) for m in self.meta_metrics]
            
            ax3.plot(meta_losses, label='Meta Loss', alpha=0.7)
            ax3.plot(adaptation_losses, label='Adaptation Loss', alpha=0.7)
            ax3.set_xlabel('Meta Episode')
            ax3.set_ylabel('Loss')
            ax3.set_title('Meta-Learning - Losses')
            ax3.legend()
            ax3.grid(True)
        
        # Adaptation metrics
        if self.adaptation_metrics:
            adaptation_rewards = [m.get('adaptation_reward', 0) for m in self.adaptation_metrics]
            ax4.plot(adaptation_rewards, alpha=0.7, color='green')
            ax4.set_xlabel('Adaptation Episode')
            ax4.set_ylabel('Adaptation Reward')
            ax4.set_title('Meta-Learning - Adaptation Rewards')
            ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "meta_learning_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved meta-learning curves: {plot_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Environment
        'env_name': 'BipedalWalker-v3',
        'env_config': {},
        
        # Training
        'num_meta_episodes': 1000,
        'max_steps_per_episode': 1000,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 10,
        
        # Algorithm
        'algorithm': 'ppo',  # 'ppo' or 'sac'
        
        # Meta-learning
        'meta_config': {
            'num_tasks': 5,
            'inner_lr': 1e-3,
            'outer_lr': 1e-4,
            'num_inner_steps': 5,
            'adaptation_episodes': 10,
            'meta_algorithm': 'maml'  # 'maml' or 'reptile'
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
    parser = argparse.ArgumentParser(description='Train agent with meta-learning for bipedal locomotion')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment name')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'], help='Training algorithm')
    parser.add_argument('--meta-episodes', type=int, default=1000, help='Number of meta-learning episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--num-tasks', type=int, default=5, help='Number of tasks for meta-learning')
    parser.add_argument('--inner-lr', type=float, default=1e-3, help='Inner loop learning rate')
    parser.add_argument('--outer-lr', type=float, default=1e-4, help='Outer loop learning rate')
    parser.add_argument('--adaptation-episodes', type=int, default=10, help='Episodes for task adaptation')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Base agent learning rate')
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
    config['num_meta_episodes'] = args.meta_episodes
    config['meta_config']['num_tasks'] = args.num_tasks
    config['meta_config']['inner_lr'] = args.inner_lr
    config['meta_config']['outer_lr'] = args.outer_lr
    config['meta_config']['adaptation_episodes'] = args.adaptation_episodes
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
    trainer = MetaLearningTrainer(config, device)
    trainer.train()
    
    # Cleanup
    trainer.env.close()
    trainer.writer.close()


if __name__ == "__main__":
    main()
