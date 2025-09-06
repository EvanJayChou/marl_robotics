"""
Training script for Population-based Training in Bipedal Locomotion

This script implements population-based training (PBT) for bipedal locomotion,
maintaining a population of agents with different hyperparameters and architectures.
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
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents import PPOAgent, SACAgent
from src.algorithms.population_training import PopulationTraining

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PopulationEnvironment:
    """
    Environment wrapper for population-based training.
    """
    
    def __init__(
        self,
        env_name: str = "BipedalWalker-v3",
        **kwargs
    ):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        logger.info(f"Initialized PopulationEnvironment: {env_name}")
        logger.info(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
    
    def reset(self):
        """Reset environment."""
        obs, _ = self.env.reset()
        return obs
    
    def step(self, action):
        """Take step in environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close environment."""
        self.env.close()


class PopulationMember:
    """
    Individual member of the population with its own agent and hyperparameters.
    """
    
    def __init__(
        self,
        member_id: int,
        obs_dim: int,
        action_dim: int,
        algorithm: str = "ppo",
        hyperparams: Dict[str, Any] = None,
        device: str = "cpu"
    ):
        self.member_id = member_id
        self.algorithm = algorithm
        self.hyperparams = hyperparams or {}
        self.device = torch.device(device)
        
        # Initialize agent with hyperparameters
        if algorithm == "ppo":
            self.agent = PPOAgent(
                agent_id=f"population_ppo_agent_{member_id}",
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type="continuous",
                device=self.device,
                **self.hyperparams
            )
        elif algorithm == "sac":
            self.agent = SACAgent(
                agent_id=f"population_sac_agent_{member_id}",
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type="continuous",
                device=self.device,
                **self.hyperparams
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        self.fitness_score = 0.0
        self.generation = 0
        
        logger.info(f"Initialized population member {member_id} with {algorithm}")
    
    def update_fitness(self, reward: float, length: int):
        """Update fitness score based on episode performance."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Calculate fitness as weighted combination of reward and length
        reward_weight = 0.8
        length_weight = 0.2
        self.fitness_score = reward_weight * reward + length_weight * length
    
    def get_average_fitness(self, window: int = 100) -> float:
        """Get average fitness over recent episodes."""
        if not self.episode_rewards:
            return 0.0
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        
        return 0.8 * avg_reward + 0.2 * avg_length
    
    def mutate_hyperparams(self, mutation_rate: float = 0.1):
        """Mutate hyperparameters for exploration."""
        for key, value in self.hyperparams.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    # Mutate float values by ±20%
                    mutation_factor = random.uniform(0.8, 1.2)
                    self.hyperparams[key] = value * mutation_factor
                elif isinstance(value, int) and key in ['hidden_dim', 'num_layers', 'batch_size']:
                    # Mutate integer values
                    if key == 'hidden_dim':
                        self.hyperparams[key] = random.choice([64, 128, 256, 512])
                    elif key == 'num_layers':
                        self.hyperparams[key] = random.choice([1, 2, 3, 4])
                    elif key == 'batch_size':
                        self.hyperparams[key] = random.choice([32, 64, 128, 256])
        
        logger.info(f"Member {self.member_id} hyperparameters mutated: {self.hyperparams}")
    
    def copy_from(self, other_member: 'PopulationMember'):
        """Copy parameters from another population member."""
        # Copy agent parameters
        if hasattr(self.agent, 'policy_net') and hasattr(other_member.agent, 'policy_net'):
            self.agent.policy_net.load_state_dict(other_member.agent.policy_net.state_dict())
        if hasattr(self.agent, 'value_net') and hasattr(other_member.agent, 'value_net'):
            self.agent.value_net.load_state_dict(other_member.agent.value_net.state_dict())
        
        # Copy hyperparameters
        self.hyperparams = other_member.hyperparams.copy()
        
        logger.info(f"Member {self.member_id} copied from member {other_member.member_id}")


class PopulationTrainer:
    """
    Trainer that implements population-based training for bipedal locomotion.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        
        # Training parameters
        self.num_generations = config.get('num_generations', 100)
        self.episodes_per_generation = config.get('episodes_per_generation', 100)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.save_frequency = config.get('save_frequency', 10)
        self.eval_frequency = config.get('eval_frequency', 5)
        self.log_frequency = config.get('log_frequency', 1)
        
        # Population parameters
        self.population_config = config.get('population_config', {})
        self.population_size = self.population_config.get('population_size', 8)
        self.elite_fraction = self.population_config.get('elite_fraction', 0.25)
        self.mutation_rate = self.population_config.get('mutation_rate', 0.1)
        self.crossover_rate = self.population_config.get('crossover_rate', 0.5)
        
        # Environment setup
        self.env_name = config.get('env_name', 'BipedalWalker-v3')
        self.env = PopulationEnvironment(env_name=self.env_name)
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Training tracking
        self.generation_fitness = []
        self.best_fitness_history = []
        self.population_diversity = []
        self.hyperparameter_evolution = []
        
        # Setup logging and checkpointing
        self._setup_logging()
        
        logger.info(f"Initialized PopulationTrainer with {self.population_size} members")
    
    def _initialize_population(self) -> List[PopulationMember]:
        """Initialize population with diverse hyperparameters."""
        population = []
        
        for i in range(self.population_size):
            # Generate diverse hyperparameters
            hyperparams = self._generate_hyperparameters()
            
            # Create population member
            member = PopulationMember(
                member_id=i,
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                algorithm=self.config.get('algorithm', 'ppo'),
                hyperparams=hyperparams,
                device=self.device
            )
            
            population.append(member)
        
        return population
    
    def _generate_hyperparameters(self) -> Dict[str, Any]:
        """Generate random hyperparameters for population member."""
        base_config = self.config.get('agent_config', {})
        
        hyperparams = {
            'learning_rate': random.uniform(1e-5, 1e-2),
            'gamma': random.uniform(0.9, 0.999),
            'hidden_dim': random.choice([64, 128, 256, 512]),
            'num_layers': random.choice([1, 2, 3, 4]),
            'batch_size': random.choice([32, 64, 128, 256]),
            'entropy_coef': random.uniform(0.001, 0.1),
            'value_loss_coef': random.uniform(0.1, 1.0),
            'clip_ratio': random.uniform(0.1, 0.3),
            'gae_lambda': random.uniform(0.9, 0.99),
            'ppo_epochs': random.choice([2, 3, 4, 5, 6]),
            'max_grad_norm': random.uniform(0.1, 1.0)
        }
        
        # Override with base config where specified
        for key, value in base_config.items():
            if key not in hyperparams:
                hyperparams[key] = value
        
        return hyperparams
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"population_training_{timestamp}"
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
        """Main population-based training loop."""
        logger.info("Starting population-based training...")
        
        for generation in range(self.num_generations):
            # Train population for one generation
            generation_fitness = self._train_generation(generation)
            
            # Store generation metrics
            self.generation_fitness.append(generation_fitness)
            best_fitness = max([member.get_average_fitness() for member in self.population])
            self.best_fitness_history.append(best_fitness)
            
            # Calculate population diversity
            diversity = self._calculate_population_diversity()
            self.population_diversity.append(diversity)
            
            # Logging
            if generation % self.log_frequency == 0:
                self._log_generation(generation, generation_fitness, best_fitness, diversity)
            
            # Evaluation
            if generation % self.eval_frequency == 0 and generation > 0:
                self._evaluate_population(generation)
            
            # Save checkpoint
            if generation % self.save_frequency == 0 and generation > 0:
                self._save_checkpoint(generation)
            
            # Population evolution
            if generation < self.num_generations - 1:  # Don't evolve after last generation
                self._evolve_population()
        
        # Final evaluation and save
        self._evaluate_population(self.num_generations - 1)
        self._save_checkpoint(self.num_generations - 1, final=True)
        
        logger.info("Population-based training completed!")
        self._plot_population_curves()
        self._analyze_best_member()
    
    def _train_generation(self, generation: int) -> List[float]:
        """Train all population members for one generation."""
        generation_fitness = []
        
        for member in self.population:
            member.generation = generation
            
            # Train member for specified episodes
            for episode in range(self.episodes_per_generation):
                episode_reward, episode_length, metrics = self._train_episode(member)
                
                # Update member fitness
                member.update_fitness(episode_reward, episode_length)
                
                if metrics:
                    member.training_metrics.append(metrics)
            
            # Get average fitness for this generation
            avg_fitness = member.get_average_fitness()
            generation_fitness.append(avg_fitness)
        
        return generation_fitness
    
    def _train_episode(self, member: PopulationMember) -> Tuple[float, int, Dict[str, float]]:
        """Train one episode for a population member."""
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        metrics = {}
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action, action_info = member.agent.select_action(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            if hasattr(member.agent, 'store_experience'):
                if 'value' in action_info and 'log_prob' in action_info:
                    # PPO agent
                    member.agent.store_experience(
                        obs=obs,
                        action=action,
                        reward=reward,
                        value=action_info['value'],
                        log_prob=action_info['log_prob'],
                        done=done
                    )
                else:
                    # SAC agent
                    member.agent.store_experience(
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
        if hasattr(member.agent, 'update'):
            update_metrics = member.agent.update({})
            if update_metrics:
                metrics.update(update_metrics)
        
        return episode_reward, episode_length, metrics
    
    def _evolve_population(self):
        """Evolve population using selection, crossover, and mutation."""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.get_average_fitness(), reverse=True)
        
        # Select elite members
        elite_size = int(self.population_size * self.elite_fraction)
        elite_members = self.population[:elite_size]
        
        # Create new population
        new_population = []
        
        # Keep elite members
        for i in range(elite_size):
            new_population.append(elite_members[i])
        
        # Generate new members through crossover and mutation
        for i in range(elite_size, self.population_size):
            # Select parents
            parent1 = random.choice(elite_members)
            parent2 = random.choice(elite_members)
            
            # Create new member
            new_member = PopulationMember(
                member_id=i,
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                algorithm=self.config.get('algorithm', 'ppo'),
                hyperparams=self._crossover_hyperparams(parent1.hyperparams, parent2.hyperparams),
                device=self.device
            )
            
            # Copy parameters from better parent
            if parent1.get_average_fitness() > parent2.get_average_fitness():
                new_member.copy_from(parent1)
            else:
                new_member.copy_from(parent2)
            
            # Mutate hyperparameters
            new_member.mutate_hyperparams(self.mutation_rate)
            
            new_population.append(new_member)
        
        self.population = new_population
        
        # Store hyperparameter evolution
        self.hyperparameter_evolution.append({
            'generation': len(self.generation_fitness),
            'hyperparams': [member.hyperparams for member in self.population]
        })
        
        logger.info(f"Population evolved. Best fitness: {max([m.get_average_fitness() for m in self.population]):.2f}")
    
    def _crossover_hyperparams(self, hyperparams1: Dict[str, Any], hyperparams2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two hyperparameter sets."""
        new_hyperparams = {}
        
        for key in hyperparams1:
            if random.random() < self.crossover_rate:
                new_hyperparams[key] = hyperparams1[key]
            else:
                new_hyperparams[key] = hyperparams2[key]
        
        return new_hyperparams
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of the population."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate diversity based on hyperparameter differences
        diversity_scores = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                member1 = self.population[i]
                member2 = self.population[j]
                
                # Calculate hyperparameter distance
                distance = 0.0
                for key in member1.hyperparams:
                    if key in member2.hyperparams:
                        if isinstance(member1.hyperparams[key], (int, float)):
                            distance += abs(member1.hyperparams[key] - member2.hyperparams[key])
                
                diversity_scores.append(distance)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _log_generation(self, generation: int, generation_fitness: List[float], best_fitness: float, diversity: float):
        """Log generation information."""
        avg_fitness = np.mean(generation_fitness)
        std_fitness = np.std(generation_fitness)
        
        logger.info(f"Generation {generation}:")
        logger.info(f"  Best Fitness: {best_fitness:.2f}")
        logger.info(f"  Avg Fitness: {avg_fitness:.2f} ± {std_fitness:.2f}")
        logger.info(f"  Population Diversity: {diversity:.2f}")
        
        # Log individual member fitness
        for i, member in enumerate(self.population):
            fitness = member.get_average_fitness()
            logger.info(f"  Member {i}: Fitness={fitness:.2f}")
        
        # TensorBoard logging
        self.writer.add_scalar('Population/BestFitness', best_fitness, generation)
        self.writer.add_scalar('Population/AvgFitness', avg_fitness, generation)
        self.writer.add_scalar('Population/FitnessStd', std_fitness, generation)
        self.writer.add_scalar('Population/Diversity', diversity, generation)
        
        # Log individual member fitness
        for i, member in enumerate(self.population):
            fitness = member.get_average_fitness()
            self.writer.add_scalar(f'MemberFitness/Member_{i}', fitness, generation)
    
    def _evaluate_population(self, generation: int, num_eval_episodes: int = 5):
        """Evaluate the best population member."""
        logger.info(f"Evaluating population at generation {generation}...")
        
        # Find best member
        best_member = max(self.population, key=lambda x: x.get_average_fitness())
        
        eval_rewards = []
        eval_lengths = []
        
        for eval_episode in range(num_eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    action, _ = best_member.agent.select_action(obs, deterministic=True)
                
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
        
        logger.info(f"Best member evaluation: Avg Reward={avg_eval_reward:.2f}, Avg Length={avg_eval_length:.2f}")
        logger.info(f"Best member hyperparams: {best_member.hyperparams}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Evaluation/BestMemberReward', avg_eval_reward, generation)
        self.writer.add_scalar('Evaluation/BestMemberLength', avg_eval_length, generation)
    
    def _save_checkpoint(self, generation: int, final: bool = False):
        """Save population checkpoint."""
        if final:
            checkpoint_dir = os.path.join(self.checkpoint_dir, "final_population")
        else:
            checkpoint_dir = os.path.join(self.checkpoint_dir, f"generation_{generation}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save each population member
        for i, member in enumerate(self.population):
            member_path = os.path.join(checkpoint_dir, f"member_{i}.pth")
            member.agent.save_checkpoint(member_path)
        
        # Save population state
        population_state = {
            'generation': generation,
            'population_size': self.population_size,
            'member_fitness': [member.get_average_fitness() for member in self.population],
            'member_hyperparams': [member.hyperparams for member in self.population],
            'generation_fitness': self.generation_fitness,
            'best_fitness_history': self.best_fitness_history,
            'population_diversity': self.population_diversity
        }
        
        state_path = os.path.join(checkpoint_dir, "population_state.json")
        with open(state_path, 'w') as f:
            json.dump(population_state, f, indent=2)
        
        logger.info(f"Saved population checkpoint: {checkpoint_dir}")
    
    def _analyze_best_member(self):
        """Analyze and report on the best population member."""
        best_member = max(self.population, key=lambda x: x.get_average_fitness())
        
        logger.info("=" * 60)
        logger.info("BEST POPULATION MEMBER ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Member ID: {best_member.member_id}")
        logger.info(f"Final Fitness: {best_member.get_average_fitness():.2f}")
        logger.info(f"Total Episodes: {len(best_member.episode_rewards)}")
        logger.info(f"Best Episode Reward: {max(best_member.episode_rewards):.2f}")
        logger.info(f"Average Episode Reward: {np.mean(best_member.episode_rewards):.2f}")
        logger.info("Hyperparameters:")
        for key, value in best_member.hyperparams.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    def _plot_population_curves(self):
        """Plot and save population training curves."""
        if not self.generation_fitness:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Fitness evolution
        generations = range(len(self.generation_fitness))
        best_fitness = [max(fitness) for fitness in self.generation_fitness]
        avg_fitness = [np.mean(fitness) for fitness in self.generation_fitness]
        std_fitness = [np.std(fitness) for fitness in self.generation_fitness]
        
        ax1.plot(generations, best_fitness, 'r-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'b-', label='Average Fitness', linewidth=2)
        ax1.fill_between(generations, 
                        np.array(avg_fitness) - np.array(std_fitness),
                        np.array(avg_fitness) + np.array(std_fitness),
                        alpha=0.3, color='blue')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Population Training - Fitness Evolution')
        ax1.legend()
        ax1.grid(True)
        
        # Population diversity
        ax2.plot(generations, self.population_diversity, 'g-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity')
        ax2.set_title('Population Training - Diversity Evolution')
        ax2.grid(True)
        
        # Individual member fitness over time
        for i, member in enumerate(self.population):
            if member.episode_rewards:
                # Calculate moving average
                window = 50
                if len(member.episode_rewards) >= window:
                    moving_avg = np.convolve(member.episode_rewards, np.ones(window)/window, mode='valid')
                    ax3.plot(moving_avg, alpha=0.7, label=f'Member {i}')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Population Training - Individual Member Performance')
        ax3.legend()
        ax3.grid(True)
        
        # Hyperparameter evolution (example: learning rate)
        if self.hyperparameter_evolution:
            lr_evolution = []
            for gen_data in self.hyperparameter_evolution:
                lrs = [hp.get('learning_rate', 0) for hp in gen_data['hyperparams']]
                lr_evolution.append(lrs)
            
            for i in range(len(lr_evolution[0])):
                member_lrs = [gen[i] for gen in lr_evolution]
                ax4.plot(member_lrs, alpha=0.7, label=f'Member {i}')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Population Training - Learning Rate Evolution')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "population_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved population training curves: {plot_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Environment
        'env_name': 'BipedalWalker-v3',
        
        # Training
        'num_generations': 100,
        'episodes_per_generation': 100,
        'max_steps_per_episode': 1000,
        'save_frequency': 10,
        'eval_frequency': 5,
        'log_frequency': 1,
        
        # Algorithm
        'algorithm': 'ppo',  # 'ppo' or 'sac'
        
        # Population parameters
        'population_config': {
            'population_size': 8,
            'elite_fraction': 0.25,
            'mutation_rate': 0.1,
            'crossover_rate': 0.5
        },
        
        # Base agent configuration (used as defaults)
        'agent_config': {
            'action_bounds': (-1.0, 1.0)
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train population of agents for bipedal locomotion')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment name')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'], help='Training algorithm')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--episodes-per-generation', type=int, default=100, help='Episodes per generation')
    parser.add_argument('--population-size', type=int, default=8, help='Population size')
    parser.add_argument('--elite-fraction', type=float, default=0.25, help='Fraction of elite members')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
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
    config['num_generations'] = args.generations
    config['episodes_per_generation'] = args.episodes_per_generation
    config['population_config']['population_size'] = args.population_size
    config['population_config']['elite_fraction'] = args.elite_fraction
    config['population_config']['mutation_rate'] = args.mutation_rate
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Create trainer and start training
    trainer = PopulationTrainer(config, device)
    trainer.train()
    
    # Cleanup
    trainer.env.close()
    trainer.writer.close()


if __name__ == "__main__":
    main()
