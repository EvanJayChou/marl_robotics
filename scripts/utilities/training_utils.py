"""
Training utilities and helper functions for Multi-Agent Bipedal Locomotion

This module provides utility functions for training, logging, monitoring,
and managing experiments.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import yaml
import shutil
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class ExperimentManager:
    """
    Manages experiments, logging, and checkpointing.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory structure
        self._create_directory_structure()
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger(f"experiment_{experiment_name}")
        self.logger.info(f"Initialized experiment: {experiment_name}")
    
    def _create_directory_structure(self):
        """Create experiment directory structure."""
        dirs = [
            "logs",
            "checkpoints",
            "results",
            "plots",
            "configs",
            "videos"
        ]
        
        for dir_name in dirs:
            os.makedirs(os.path.join(self.experiment_dir, dir_name), exist_ok=True)
    
    def _setup_logging(self):
        """Setup experiment-specific logging."""
        log_file = os.path.join(self.experiment_dir, "logs", f"experiment_{self.timestamp}.log")
        
        # Create logger
        logger = logging.getLogger(f"experiment_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def save_config(self, config: Dict[str, Any], config_name: str = "config"):
        """Save experiment configuration."""
        config_path = os.path.join(self.experiment_dir, "configs", f"{config_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Saved configuration: {config_path}")
        return config_path
    
    def save_checkpoint(self, agent, episode: int, is_best: bool = False):
        """Save agent checkpoint."""
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth")
        
        agent.save_checkpoint(checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def save_results(self, results: Dict[str, Any], results_name: str = "results"):
        """Save experiment results."""
        results_path = os.path.join(self.experiment_dir, "results", f"{results_name}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results: {results_path}")
        return results_path
    
    def save_plot(self, fig, plot_name: str):
        """Save plot figure."""
        plot_path = os.path.join(self.experiment_dir, "plots", f"{plot_name}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved plot: {plot_path}")
        return plot_path
    
    def get_experiment_path(self, subdir: str = ""):
        """Get path within experiment directory."""
        if subdir:
            return os.path.join(self.experiment_dir, subdir)
        return self.experiment_dir


class PerformanceMonitor:
    """
    Monitors and tracks training performance metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.history = {}
    
    def update(self, episode: int, metrics: Dict[str, float]):
        """Update metrics for an episode."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []
            
            self.metrics[key].append(value)
            self.history[key].append((episode, value))
    
    def get_moving_average(self, metric_name: str) -> float:
        """Get moving average of a metric."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if len(values) < self.window_size:
            return np.mean(values)
        
        return np.mean(values[-self.window_size:])
    
    def get_best_value(self, metric_name: str, higher_is_better: bool = True) -> Tuple[float, int]:
        """Get best value and episode for a metric."""
        if metric_name not in self.metrics:
            return 0.0, 0
        
        values = self.metrics[metric_name]
        if not values:
            return 0.0, 0
        
        if higher_is_better:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return values[best_idx], best_idx
    
    def get_improvement_rate(self, metric_name: str) -> float:
        """Calculate improvement rate over recent episodes."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if len(values) < 2:
            return 0.0
        
        # Compare recent performance to earlier performance
        recent_window = min(self.window_size // 2, len(values) // 2)
        if recent_window < 2:
            return 0.0
        
        recent_avg = np.mean(values[-recent_window:])
        earlier_avg = np.mean(values[-2*recent_window:-recent_window])
        
        if earlier_avg == 0:
            return 0.0
        
        return (recent_avg - earlier_avg) / abs(earlier_avg)
    
    def should_early_stop(self, metric_name: str, patience: int = 50, min_delta: float = 0.01) -> bool:
        """Check if training should stop early based on metric improvement."""
        if metric_name not in self.metrics:
            return False
        
        values = self.metrics[metric_name]
        if len(values) < patience:
            return False
        
        # Check if metric has improved significantly in the last 'patience' episodes
        recent_values = values[-patience:]
        if len(recent_values) < 2:
            return False
        
        improvement = max(recent_values) - min(recent_values)
        return improvement < min_delta
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for metric_name in self.metrics:
            values = self.metrics[metric_name]
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'moving_avg': self.get_moving_average(metric_name),
                    'improvement_rate': self.get_improvement_rate(metric_name)
                }
        
        return summary


class HyperparameterScheduler:
    """
    Schedules hyperparameters during training.
    """
    
    def __init__(self, initial_params: Dict[str, Any]):
        self.initial_params = initial_params.copy()
        self.current_params = initial_params.copy()
        self.schedules = {}
    
    def add_schedule(self, param_name: str, schedule_type: str, **kwargs):
        """Add a hyperparameter schedule."""
        self.schedules[param_name] = {
            'type': schedule_type,
            'kwargs': kwargs
        }
    
    def update(self, episode: int):
        """Update hyperparameters based on current episode."""
        for param_name, schedule in self.schedules.items():
            if param_name in self.initial_params:
                self.current_params[param_name] = self._calculate_value(
                    self.initial_params[param_name],
                    episode,
                    schedule['type'],
                    **schedule['kwargs']
                )
    
    def _calculate_value(self, initial_value: float, episode: int, schedule_type: str, **kwargs) -> float:
        """Calculate parameter value based on schedule type."""
        if schedule_type == 'linear':
            start_value = kwargs.get('start_value', initial_value)
            end_value = kwargs.get('end_value', initial_value * 0.1)
            total_episodes = kwargs.get('total_episodes', 1000)
            
            if episode >= total_episodes:
                return end_value
            
            progress = episode / total_episodes
            return start_value + (end_value - start_value) * progress
        
        elif schedule_type == 'exponential':
            decay_rate = kwargs.get('decay_rate', 0.99)
            return initial_value * (decay_rate ** episode)
        
        elif schedule_type == 'cosine':
            min_value = kwargs.get('min_value', initial_value * 0.1)
            total_episodes = kwargs.get('total_episodes', 1000)
            
            if episode >= total_episodes:
                return min_value
            
            progress = episode / total_episodes
            return min_value + (initial_value - min_value) * 0.5 * (1 + np.cos(np.pi * progress))
        
        elif schedule_type == 'step':
            step_size = kwargs.get('step_size', 100)
            decay_factor = kwargs.get('decay_factor', 0.5)
            
            steps = episode // step_size
            return initial_value * (decay_factor ** steps)
        
        else:
            return initial_value
    
    def get_current_params(self) -> Dict[str, Any]:
        """Get current hyperparameter values."""
        return self.current_params.copy()


class DataLogger:
    """
    Logs training data and metrics.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.data = {}
        self.episode_data = []
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log episode data."""
        episode_entry = {'episode': episode, **metrics}
        self.episode_data.append(episode_entry)
        
        # Update running data
        for key, value in metrics.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
    
    def save_data(self, filename: str = "training_data.json"):
        """Save logged data to file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
    
    def plot_metrics(self, metrics: List[str], save_path: str = None):
        """Plot specified metrics."""
        if not metrics:
            return
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in self.data:
                axes[i].plot(self.data[metric])
                axes[i].set_title(f'{metric} Over Episodes')
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class ModelComparator:
    """
    Compares different trained models.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, checkpoint_path: str, config: Dict[str, Any] = None):
        """Add a model for comparison."""
        self.models[name] = {
            'checkpoint_path': checkpoint_path,
            'config': config or {}
        }
    
    def compare_models(self, evaluator_func, **kwargs) -> Dict[str, Any]:
        """Compare all added models using the provided evaluator function."""
        comparison_results = {}
        
        for name, model_info in self.models.items():
            print(f"Evaluating model: {name}")
            results = evaluator_func(model_info['checkpoint_path'], **kwargs)
            comparison_results[name] = results
        
        self.results = comparison_results
        return comparison_results
    
    def generate_comparison_report(self, output_dir: str):
        """Generate a comprehensive comparison report."""
        if not self.results:
            print("No results to compare. Run compare_models first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison plots
        self._plot_comparison(output_dir)
        
        # Create summary table
        self._create_summary_table(output_dir)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Comparison report saved to: {output_dir}")
    
    def _plot_comparison(self, output_dir: str):
        """Create comparison plots."""
        metrics = ['episode_rewards', 'episode_lengths']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            for name, results in self.results.items():
                if metric in results:
                    values = results[metric]
                    plt.plot(values, label=name, alpha=0.7)
            
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xlabel('Episode')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, f"{metric}_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_summary_table(self, output_dir: str):
        """Create a summary table of model performance."""
        summary_data = []
        
        for name, results in self.results.items():
            if 'statistics' in results:
                stats = results['statistics']
                summary_data.append({
                    'Model': name,
                    'Mean Reward': f"{stats['rewards']['mean']:.2f}",
                    'Std Reward': f"{stats['rewards']['std']:.2f}",
                    'Success Rate': f"{stats['success_rate']:.3f}",
                    'Stability': f"{stats['stability']:.3f}",
                    'Efficiency': f"{stats['efficiency']:.3f}"
                })
        
        # Save as JSON
        summary_path = os.path.join(output_dir, "summary_table.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Print table
        print("\nModel Comparison Summary:")
        print("-" * 80)
        for row in summary_data:
            print(f"{row['Model']:20} | {row['Mean Reward']:10} | {row['Success Rate']:10} | {row['Stability']:10}")


def create_training_config(
    algorithm: str = "ppo",
    env_name: str = "BipedalWalker-v3",
    num_episodes: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standard training configuration.
    
    Args:
        algorithm: Training algorithm ('ppo', 'sac', 'multi_agent_ppo')
        env_name: Environment name
        num_episodes: Number of training episodes
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing training configuration
    """
    base_config = {
        'env_name': env_name,
        'num_episodes': num_episodes,
        'max_steps_per_episode': 1000,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 10,
        'device': 'auto'
    }
    
    if algorithm == "ppo":
        base_config.update({
            'algorithm': 'ppo',
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
        })
    elif algorithm == "sac":
        base_config.update({
            'algorithm': 'sac',
            'agent_config': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'automatic_entropy_tuning': True,
                'hidden_dim': 256,
                'num_layers': 2,
                'buffer_size': 1000000,
                'batch_size': 256,
                'action_bounds': (-1.0, 1.0)
            }
        })
    elif algorithm == "multi_agent_ppo":
        base_config.update({
            'algorithm': 'multi_agent_ppo',
            'num_agents': 2,
            'communication_type': 'attention',
            'coordination_type': 'centralized',
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
                'num_attention_heads': 4
            }
        })
    
    # Override with provided kwargs
    base_config.update(kwargs)
    
    return base_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file."""
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get appropriate device for training."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_training_info(config: Dict[str, Any], device: torch.device):
    """Print training configuration information."""
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {config.get('env_name', 'Unknown')}")
    print(f"Algorithm: {config.get('algorithm', 'Unknown')}")
    print(f"Episodes: {config.get('num_episodes', 'Unknown')}")
    print(f"Device: {device}")
    print(f"Max Steps per Episode: {config.get('max_steps_per_episode', 'Unknown')}")
    
    if 'agent_config' in config:
        agent_config = config['agent_config']
        print(f"Learning Rate: {agent_config.get('learning_rate', 'Unknown')}")
        print(f"Batch Size: {agent_config.get('batch_size', 'Unknown')}")
        print(f"Hidden Dimension: {agent_config.get('hidden_dim', 'Unknown')}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    config = create_training_config("ppo", num_episodes=500)
    print("Example training configuration:")
    print(json.dumps(config, indent=2))
