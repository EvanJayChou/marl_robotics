"""
Evaluation script for trained agents in Bipedal Locomotion

This script evaluates trained agents and generates comprehensive performance reports.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import gym
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents import PPOAgent, SACAgent, MultiAgentPPO, AttentionAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Comprehensive evaluator for trained agents.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        env_name: str = "BipedalWalker-v3",
        device: str = "cpu",
        render: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.env_name = env_name
        self.device = torch.device(device)
        self.render = render
        
        # Load agent and environment
        self.agent = self._load_agent()
        self.env = gym.make(env_name, render_mode='human' if render else None)
        
        # Evaluation results
        self.evaluation_results = {}
        
        logger.info(f"Initialized AgentEvaluator for {env_name}")
        logger.info(f"Agent type: {type(self.agent).__name__}")
    
    def _load_agent(self):
        """Load agent from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Determine agent type from checkpoint
        if 'agent_id' in checkpoint:
            agent_id = checkpoint['agent_id']
        else:
            agent_id = "evaluated_agent"
        
        # Get configuration
        config = checkpoint.get('config', {})
        obs_dim = config.get('obs_dim', 28)
        action_dim = config.get('action_dim', 6)
        action_type = config.get('action_type', 'continuous')
        
        # Create appropriate agent
        if 'policy_net_state_dict' in checkpoint and 'value_net_state_dict' in checkpoint:
            # PPO agent
            agent = PPOAgent(
                agent_id=agent_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=action_type,
                device=self.device,
                **config
            )
        elif 'actor_state_dict' in checkpoint and 'critic1_state_dict' in checkpoint:
            # SAC agent
            agent = SACAgent(
                agent_id=agent_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=action_type,
                device=self.device,
                **config
            )
        elif 'agent_ids' in checkpoint:
            # Multi-agent PPO
            agent_ids = checkpoint['agent_ids']
            num_agents = checkpoint['num_agents']
            agent = MultiAgentPPO(
                agent_ids=agent_ids,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=action_type,
                device=self.device,
                num_agents=num_agents,
                **config
            )
        else:
            # Default to PPO
            agent = PPOAgent(
                agent_id=agent_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=action_type,
                device=self.device,
                **config
            )
        
        # Load state
        agent.load_checkpoint(self.checkpoint_path)
        agent.set_training_mode(False)
        
        return agent
    
    def evaluate(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
        save_videos: bool = False,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            save_videos: Whether to save episode videos
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation with {num_episodes} episodes...")
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"evaluation_results_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        episode_info = []
        
        for episode in range(num_episodes):
            reward, length, info = self._run_episode(deterministic)
            episode_rewards.append(reward)
            episode_lengths.append(length)
            episode_info.append(info)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}")
        
        # Calculate statistics
        stats = self._calculate_statistics(episode_rewards, episode_lengths, episode_info)
        
        # Store results
        self.evaluation_results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_info': episode_info,
            'statistics': stats,
            'checkpoint_path': self.checkpoint_path,
            'env_name': self.env_name,
            'num_episodes': num_episodes,
            'deterministic': deterministic
        }
        
        # Save results
        self._save_results(output_dir)
        
        # Generate plots
        self._generate_plots(output_dir)
        
        # Print summary
        self._print_summary()
        
        return self.evaluation_results
    
    def _run_episode(self, deterministic: bool = True) -> Tuple[float, int, Dict[str, Any]]:
        """Run a single evaluation episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = {
            'actions': [],
            'rewards': [],
            'observations': [],
            'done_reason': None
        }
        
        for step in range(1000):  # Max steps
            # Select action
            if hasattr(self.agent, 'select_action'):
                action, action_info = self.agent.select_action(obs, deterministic=deterministic)
            else:
                # For multi-agent systems
                action, action_info = self.agent.select_action(obs.reshape(1, -1), deterministic=deterministic)
                action = action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store episode information
            episode_info['actions'].append(action.copy())
            episode_info['rewards'].append(reward)
            episode_info['observations'].append(obs.copy())
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                episode_info['done_reason'] = 'terminated' if terminated else 'truncated'
                break
        
        return episode_reward, episode_length, episode_info
    
    def _calculate_statistics(
        self,
        episode_rewards: List[float],
        episode_lengths: List[int],
        episode_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics."""
        stats = {
            'rewards': {
                'mean': np.mean(episode_rewards),
                'std': np.std(episode_rewards),
                'min': np.min(episode_rewards),
                'max': np.max(episode_rewards),
                'median': np.median(episode_rewards),
                'q25': np.percentile(episode_rewards, 25),
                'q75': np.percentile(episode_rewards, 75)
            },
            'lengths': {
                'mean': np.mean(episode_lengths),
                'std': np.std(episode_lengths),
                'min': np.min(episode_lengths),
                'max': np.max(episode_lengths),
                'median': np.median(episode_lengths),
                'q25': np.percentile(episode_lengths, 25),
                'q75': np.percentile(episode_lengths, 75)
            },
            'success_rate': self._calculate_success_rate(episode_rewards, episode_lengths),
            'stability': self._calculate_stability(episode_rewards),
            'efficiency': self._calculate_efficiency(episode_rewards, episode_lengths)
        }
        
        return stats
    
    def _calculate_success_rate(self, rewards: List[float], lengths: List[int]) -> float:
        """Calculate success rate based on reward and length thresholds."""
        # Define success criteria
        min_reward = 200  # Minimum reward for success
        min_length = 100  # Minimum episode length for success
        
        successful_episodes = sum(1 for r, l in zip(rewards, lengths) 
                                if r >= min_reward and l >= min_length)
        
        return successful_episodes / len(rewards)
    
    def _calculate_stability(self, rewards: List[float]) -> float:
        """Calculate stability as inverse of coefficient of variation."""
        if np.mean(rewards) == 0:
            return 0.0
        
        cv = np.std(rewards) / np.mean(rewards)
        return 1.0 / (1.0 + cv)  # Normalize to [0, 1]
    
    def _calculate_efficiency(self, rewards: List[float], lengths: List[int]) -> float:
        """Calculate efficiency as reward per step."""
        total_reward = sum(rewards)
        total_steps = sum(lengths)
        
        return total_reward / total_steps if total_steps > 0 else 0.0
    
    def _save_results(self, output_dir: str):
        """Save evaluation results to files."""
        # Save main results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results = self.evaluation_results.copy()
            results['episode_rewards'] = [float(r) for r in results['episode_rewards']]
            results['episode_lengths'] = [int(l) for l in results['episode_lengths']]
            
            # Remove large arrays from episode_info for JSON
            for info in results['episode_info']:
                if 'actions' in info:
                    del info['actions']
                if 'observations' in info:
                    del info['observations']
            
            json.dump(results, f, indent=2)
        
        # Save detailed episode data
        detailed_path = os.path.join(output_dir, "detailed_episode_data.npz")
        np.savez(detailed_path,
                episode_rewards=np.array(self.evaluation_results['episode_rewards']),
                episode_lengths=np.array(self.evaluation_results['episode_lengths']))
        
        logger.info(f"Saved results to {output_dir}")
    
    def _generate_plots(self, output_dir: str):
        """Generate evaluation plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episode_rewards = self.evaluation_results['episode_rewards']
        episode_lengths = self.evaluation_results['episode_lengths']
        stats = self.evaluation_results['statistics']
        
        # Reward distribution
        ax1.hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(stats['rewards']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {stats['rewards']['mean']:.2f}")
        ax1.axvline(stats['rewards']['median'], color='green', linestyle='--', 
                   label=f"Median: {stats['rewards']['median']:.2f}")
        ax1.set_xlabel('Episode Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Episode Reward Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode length distribution
        ax2.hist(episode_lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(stats['lengths']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {stats['lengths']['mean']:.1f}")
        ax2.axvline(stats['lengths']['median'], color='blue', linestyle='--', 
                   label=f"Median: {stats['lengths']['median']:.1f}")
        ax2.set_xlabel('Episode Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Episode Length Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reward over episodes
        ax3.plot(episode_rewards, alpha=0.7, color='blue')
        ax3.axhline(stats['rewards']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {stats['rewards']['mean']:.2f}")
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Episode Rewards Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics
        metrics = ['Success Rate', 'Stability', 'Efficiency']
        values = [stats['success_rate'], stats['stability'], stats['efficiency']]
        colors = ['green', 'orange', 'purple']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Metrics')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "evaluation_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plots to {plot_path}")
    
    def _print_summary(self):
        """Print evaluation summary."""
        stats = self.evaluation_results['statistics']
        
        print("\n" + "="*60)
        print("AGENT EVALUATION SUMMARY")
        print("="*60)
        print(f"Environment: {self.env_name}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Episodes: {self.evaluation_results['num_episodes']}")
        print(f"Deterministic: {self.evaluation_results['deterministic']}")
        print()
        
        print("REWARD STATISTICS:")
        print(f"  Mean: {stats['rewards']['mean']:.2f} ± {stats['rewards']['std']:.2f}")
        print(f"  Median: {stats['rewards']['median']:.2f}")
        print(f"  Range: [{stats['rewards']['min']:.2f}, {stats['rewards']['max']:.2f}]")
        print(f"  Q25-Q75: [{stats['rewards']['q25']:.2f}, {stats['rewards']['q75']:.2f}]")
        print()
        
        print("EPISODE LENGTH STATISTICS:")
        print(f"  Mean: {stats['lengths']['mean']:.1f} ± {stats['lengths']['std']:.1f}")
        print(f"  Median: {stats['lengths']['median']:.1f}")
        print(f"  Range: [{stats['lengths']['min']}, {stats['lengths']['max']}]")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"  Success Rate: {stats['success_rate']:.3f}")
        print(f"  Stability: {stats['stability']:.3f}")
        print(f"  Efficiency: {stats['efficiency']:.3f}")
        print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate trained agent for bipedal locomotion')
    parser.add_argument('checkpoint', type=str, help='Path to agent checkpoint')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment name')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions (overrides deterministic)')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Deterministic vs stochastic
    deterministic = args.deterministic and not args.stochastic
    
    logger.info(f"Using device: {device}")
    logger.info(f"Deterministic actions: {deterministic}")
    
    # Create evaluator and run evaluation
    evaluator = AgentEvaluator(
        checkpoint_path=args.checkpoint,
        env_name=args.env,
        device=device,
        render=args.render
    )
    
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        deterministic=deterministic,
        output_dir=args.output_dir
    )
    
    # Cleanup
    evaluator.env.close()


if __name__ == "__main__":
    main()
