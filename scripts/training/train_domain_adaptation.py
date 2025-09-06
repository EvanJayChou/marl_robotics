"""
Training script for Domain Adaptation in Bipedal Locomotion

This script implements domain adaptation techniques for transferring
learned policies from simulation to real-world environments (Sim2Real).
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
from src.algorithms.domain_adaptation import DomainAdaptation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Sim2RealEnvironment:
    """
    Environment wrapper that supports domain adaptation between simulation and real-world.
    """
    
    def __init__(
        self,
        sim_env_name: str = "BipedalWalker-v3",
        real_env_name: str = None,
        domain_config: Dict[str, Any] = None,
        **kwargs
    ):
        self.sim_env_name = sim_env_name
        self.real_env_name = real_env_name
        self.domain_config = domain_config or {}
        
        # Initialize simulation environment
        self.sim_env = gym.make(sim_env_name)
        self.obs_dim = self.sim_env.observation_space.shape[0]
        self.action_dim = self.sim_env.action_space.shape[0]
        
        # Real environment (placeholder - would be actual robot interface)
        self.real_env = None
        if real_env_name:
            # In a real implementation, this would connect to actual robot
            logger.info(f"Real environment {real_env_name} would be initialized here")
        
        # Domain adaptation parameters
        self.current_domain = 'sim'  # 'sim' or 'real'
        self.domain_shift_params = self._create_domain_shift_params()
        
        logger.info(f"Initialized Sim2RealEnvironment: {sim_env_name} -> {real_env_name}")
    
    def _create_domain_shift_params(self) -> Dict[str, Any]:
        """Create parameters for domain shift simulation."""
        return {
            'sim': {
                'noise_level': 0.0,
                'delay': 0,
                'actuator_dynamics': 'ideal',
                'sensor_noise': 0.0,
                'friction': 1.0,
                'gravity': 9.8
            },
            'real': {
                'noise_level': 0.1,
                'delay': 1,
                'actuator_dynamics': 'realistic',
                'sensor_noise': 0.05,
                'friction': 0.8,
                'gravity': 9.8
            }
        }
    
    def set_domain(self, domain: str):
        """Set the current domain (sim or real)."""
        if domain in ['sim', 'real']:
            self.current_domain = domain
            logger.info(f"Switched to {domain} domain")
        else:
            logger.warning(f"Invalid domain: {domain}")
    
    def get_domain_params(self) -> Dict[str, Any]:
        """Get parameters for current domain."""
        return self.domain_shift_params[self.current_domain].copy()
    
    def reset(self):
        """Reset environment."""
        if self.current_domain == 'sim':
            obs, _ = self.sim_env.reset()
        else:
            # In real implementation, reset real robot
            obs = np.random.randn(self.obs_dim)  # Placeholder
        
        # Apply domain-specific modifications
        domain_params = self.get_domain_params()
        obs = self._apply_domain_shift(obs, domain_params)
        
        return obs
    
    def step(self, action):
        """Take step in environment."""
        domain_params = self.get_domain_params()
        
        # Apply domain-specific action modifications
        action = self._apply_action_shift(action, domain_params)
        
        if self.current_domain == 'sim':
            obs, reward, terminated, truncated, info = self.sim_env.step(action)
        else:
            # In real implementation, send action to real robot
            obs = np.random.randn(self.obs_dim)  # Placeholder
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
        
        # Apply domain-specific observation modifications
        obs = self._apply_domain_shift(obs, domain_params)
        
        # Add domain information to info
        info['domain'] = self.current_domain
        info['domain_params'] = domain_params
        
        return obs, reward, terminated, truncated, info
    
    def _apply_domain_shift(self, obs: np.ndarray, domain_params: Dict[str, Any]) -> np.ndarray:
        """Apply domain shift to observations."""
        # Add sensor noise
        noise_level = domain_params.get('sensor_noise', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, obs.shape)
            obs = obs + noise
        
        return obs
    
    def _apply_action_shift(self, action: np.ndarray, domain_params: Dict[str, Any]) -> np.ndarray:
        """Apply domain shift to actions."""
        # Add actuator noise
        noise_level = domain_params.get('noise_level', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, action.shape)
            action = action + noise
        
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def close(self):
        """Close environment."""
        if self.sim_env:
            self.sim_env.close()
        if self.real_env:
            # Close real robot connection
            pass


class DomainAdaptationTrainer:
    """
    Trainer that implements domain adaptation for Sim2Real transfer.
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
        
        # Domain adaptation parameters
        self.adaptation_config = config.get('adaptation_config', {})
        self.sim_episodes = self.adaptation_config.get('sim_episodes', 800)
        self.adaptation_episodes = self.adaptation_config.get('adaptation_episodes', 200)
        self.domain_mixing_ratio = self.adaptation_config.get('domain_mixing_ratio', 0.5)
        
        # Environment setup
        self.sim_env_name = config.get('sim_env_name', 'BipedalWalker-v3')
        self.real_env_name = config.get('real_env_name', None)
        self.env = Sim2RealEnvironment(
            sim_env_name=self.sim_env_name,
            real_env_name=self.real_env_name,
            **config.get('env_config', {})
        )
        
        # Initialize domain adaptation
        self.domain_adaptation = DomainAdaptation(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            **self.adaptation_config
        )
        
        # Initialize agent
        algorithm = config.get('algorithm', 'ppo')
        if algorithm == 'ppo':
            self.agent = PPOAgent(
                agent_id="domain_adaptation_ppo_agent",
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                action_type="continuous",
                device=self.device,
                **config.get('agent_config', {})
            )
        elif algorithm == 'sac':
            self.agent = SACAgent(
                agent_id="domain_adaptation_sac_agent",
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
        self.domain_types = []
        self.training_metrics = []
        self.adaptation_metrics = []
        
        # Setup logging and checkpointing
        self._setup_logging()
        
        logger.info(f"Initialized DomainAdaptationTrainer with {algorithm} agent")
        logger.info(f"Sim episodes: {self.sim_episodes}, Adaptation episodes: {self.adaptation_episodes}")
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"domain_adaptation_{timestamp}"
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
        """Main training loop with domain adaptation."""
        logger.info("Starting domain adaptation training...")
        
        # Phase 1: Train in simulation
        logger.info("Phase 1: Training in simulation...")
        self._train_simulation_phase()
        
        # Phase 2: Domain adaptation
        logger.info("Phase 2: Domain adaptation...")
        self._train_adaptation_phase()
        
        # Final evaluation
        self._evaluate_final()
        self._save_checkpoint(self.num_episodes - 1, final=True)
        
        logger.info("Domain adaptation training completed!")
        self._plot_training_curves()
        self._plot_domain_adaptation_curves()
    
    def _train_simulation_phase(self):
        """Train agent in simulation environment."""
        self.env.set_domain('sim')
        
        for episode in range(self.sim_episodes):
            episode_reward, episode_length, metrics = self._train_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.domain_types.append('sim')
            if metrics:
                self.training_metrics.append(metrics)
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_episode(episode, episode_reward, episode_length, metrics, 'sim')
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                self._evaluate(episode, 'sim')
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode, phase='sim')
    
    def _train_adaptation_phase(self):
        """Train agent with domain adaptation."""
        for episode in range(self.sim_episodes, self.num_episodes):
            # Determine domain for this episode
            if np.random.random() < self.domain_mixing_ratio:
                domain = 'sim'
            else:
                domain = 'real'
            
            self.env.set_domain(domain)
            
            episode_reward, episode_length, metrics = self._train_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.domain_types.append(domain)
            if metrics:
                self.training_metrics.append(metrics)
            
            # Domain adaptation update
            adaptation_metrics = self.domain_adaptation.update(
                obs=self._get_recent_observations(),
                actions=self._get_recent_actions(),
                domain=domain
            )
            if adaptation_metrics:
                self.adaptation_metrics.append(adaptation_metrics)
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_episode(episode, episode_reward, episode_length, metrics, domain)
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                self._evaluate(episode, domain)
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode, phase='adaptation')
    
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
    
    def _get_recent_observations(self) -> np.ndarray:
        """Get recent observations for domain adaptation."""
        # In a real implementation, this would return recent observations
        return np.random.randn(10, self.env.obs_dim)  # Placeholder
    
    def _get_recent_actions(self) -> np.ndarray:
        """Get recent actions for domain adaptation."""
        # In a real implementation, this would return recent actions
        return np.random.randn(10, self.env.action_dim)  # Placeholder
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float], domain: str):
        """Log episode information."""
        # Console logging
        logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}, Domain={domain}")
        if metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(f"  Metrics: {metric_str}")
        
        # TensorBoard logging
        self.writer.add_scalar('Episode/Reward', reward, episode)
        self.writer.add_scalar('Episode/Length', length, episode)
        self.writer.add_scalar('Domain/Type', 1 if domain == 'real' else 0, episode)
        
        if metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Training/{key}', value, episode)
        
        # Moving averages
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/AvgReward_100', avg_reward, episode)
            self.writer.add_scalar('Episode/AvgLength_100', avg_length, episode)
    
    def _evaluate(self, episode: int, domain: str, num_eval_episodes: int = 5):
        """Evaluate the agent."""
        logger.info(f"Evaluating at episode {episode} in {domain} domain...")
        
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
        
        logger.info(f"Evaluation ({domain}): Avg Reward={avg_eval_reward:.2f}, Avg Length={avg_eval_length:.2f}")
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Evaluation/{domain}/AvgReward', avg_eval_reward, episode)
        self.writer.add_scalar(f'Evaluation/{domain}/AvgLength', avg_eval_length, episode)
    
    def _evaluate_final(self):
        """Final evaluation in both domains."""
        logger.info("Final evaluation...")
        
        # Evaluate in simulation
        self.env.set_domain('sim')
        self._evaluate(self.num_episodes - 1, 'sim', num_eval_episodes=10)
        
        # Evaluate in real domain (if available)
        if self.real_env_name:
            self.env.set_domain('real')
            self._evaluate(self.num_episodes - 1, 'real', num_eval_episodes=5)
    
    def _save_checkpoint(self, episode: int, phase: str = None, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            phase_suffix = f"_{phase}" if phase else ""
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}{phase_suffix}.pth")
        
        self.agent.save_checkpoint(checkpoint_path)
        
        # Save domain adaptation state
        adaptation_path = checkpoint_path.replace('.pth', '_adaptation.json')
        adaptation_state = {
            'episode': episode,
            'phase': phase,
            'domain_types': self.domain_types[-100:],  # Last 100 episodes
            'adaptation_metrics': self.adaptation_metrics[-10:] if self.adaptation_metrics else []
        }
        with open(adaptation_path, 'w') as f:
            json.dump(adaptation_state, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Saved adaptation state: {adaptation_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        if not self.episode_rewards:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward curve with domain coloring
        sim_episodes = [i for i, d in enumerate(self.domain_types) if d == 'sim']
        real_episodes = [i for i, d in enumerate(self.domain_types) if d == 'real']
        
        ax1.plot(sim_episodes, [self.episode_rewards[i] for i in sim_episodes], 
                'b.', alpha=0.6, label='Simulation')
        ax1.plot(real_episodes, [self.episode_rewards[i] for i in real_episodes], 
                'r.', alpha=0.6, label='Real')
        
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_rewards)), moving_avg, 'k-', linewidth=2, label='Moving Avg')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Domain Adaptation - Training Rewards')
        ax1.legend()
        ax1.grid(True)
        
        # Domain progression
        domain_numeric = [1 if d == 'real' else 0 for d in self.domain_types]
        ax2.plot(domain_numeric, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Domain (0=Sim, 1=Real)')
        ax2.set_title('Domain Adaptation - Domain Progression')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True)
        
        # Episode length curve
        ax3.plot(self.episode_lengths, alpha=0.3, color='orange')
        if len(self.episode_lengths) >= 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            ax3.plot(range(99, len(self.episode_lengths)), moving_avg, color='red', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Domain Adaptation - Episode Lengths')
        ax3.grid(True)
        
        # Training metrics
        if self.training_metrics:
            policy_losses = [m.get('policy_loss', 0) for m in self.training_metrics]
            value_losses = [m.get('value_loss', 0) for m in self.training_metrics]
            
            ax4.plot(policy_losses, label='Policy Loss', alpha=0.7)
            ax4.plot(value_losses, label='Value Loss', alpha=0.7)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.set_title('Domain Adaptation - Training Losses')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves: {plot_path}")
    
    def _plot_domain_adaptation_curves(self):
        """Plot domain adaptation specific curves."""
        if not self.adaptation_metrics:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot adaptation metrics
        episodes = range(len(self.adaptation_metrics))
        domain_losses = [m.get('domain_loss', 0) for m in self.adaptation_metrics]
        adaptation_losses = [m.get('adaptation_loss', 0) for m in self.adaptation_metrics]
        
        ax.plot(episodes, domain_losses, label='Domain Loss', alpha=0.7)
        ax.plot(episodes, adaptation_losses, label='Adaptation Loss', alpha=0.7)
        
        ax.set_xlabel('Adaptation Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Domain Adaptation - Adaptation Losses')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "domain_adaptation_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved domain adaptation curves: {plot_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Environment
        'sim_env_name': 'BipedalWalker-v3',
        'real_env_name': None,  # Would be actual robot interface
        'env_config': {},
        
        # Training
        'num_episodes': 1000,
        'max_steps_per_episode': 1000,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 10,
        
        # Algorithm
        'algorithm': 'ppo',  # 'ppo' or 'sac'
        
        # Domain adaptation
        'adaptation_config': {
            'sim_episodes': 800,
            'adaptation_episodes': 200,
            'domain_mixing_ratio': 0.5,
            'adaptation_lr': 1e-4,
            'domain_loss_weight': 1.0,
            'adaptation_loss_weight': 0.1
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
    parser = argparse.ArgumentParser(description='Train agent with domain adaptation for Sim2Real transfer')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--sim-env', type=str, default='BipedalWalker-v3', help='Simulation environment name')
    parser.add_argument('--real-env', type=str, help='Real environment name')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'], help='Training algorithm')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--sim-episodes', type=int, default=800, help='Episodes in simulation')
    parser.add_argument('--adaptation-episodes', type=int, default=200, help='Episodes for adaptation')
    parser.add_argument('--domain-mixing-ratio', type=float, default=0.5, help='Ratio of real domain episodes')
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
    config['sim_env_name'] = args.sim_env
    config['real_env_name'] = args.real_env
    config['algorithm'] = args.algorithm
    config['num_episodes'] = args.episodes
    config['adaptation_config']['sim_episodes'] = args.sim_episodes
    config['adaptation_config']['adaptation_episodes'] = args.adaptation_episodes
    config['adaptation_config']['domain_mixing_ratio'] = args.domain_mixing_ratio
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
    trainer = DomainAdaptationTrainer(config, device)
    trainer.train()
    
    # Cleanup
    trainer.env.close()
    trainer.writer.close()


if __name__ == "__main__":
    main()
