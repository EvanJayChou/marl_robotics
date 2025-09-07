"""
Weights & Biases (WandB) interface for experiment tracking and visualization.
"""

import wandb
from typing import Dict, List, Optional, Union
import numpy as np
import os
import json

class WandBInterface:
    def __init__(self, config: dict):
        """
        Initialize WandB interface for experiment tracking.
        
        Args:
            config (dict): Configuration for WandB logging
        """
        self.config = config
        self.run = None
        self._setup_wandb()
        
    def _setup_wandb(self):
        """Initialize WandB run."""
        project_name = self.config.get("wandb_project", "marl_robotics")
        experiment_name = self.config.get("experiment_name", "default")
        
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=self.config,
            resume="allow"
        )
        
    def log_metrics(self, metrics: Dict[str, Union[float, int, np.ndarray]]):
        """
        Log training metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        # Process numpy arrays and tensors
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                processed_metrics[key] = value.tolist()
            else:
                processed_metrics[key] = value
                
        wandb.log(processed_metrics)
        
    def log_episode(
        self,
        episode_metrics: Dict[str, float],
        episode_num: int
    ):
        """
        Log episode-level metrics.
        
        Args:
            episode_metrics: Metrics for the episode
            episode_num: Episode number
        """
        metrics = {
            "episode": episode_num,
            **episode_metrics
        }
        self.log_metrics(metrics)
        
    def log_model_gradients(
        self,
        model,
        step: int
    ):
        """
        Log model gradient statistics.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                gradient_stats.update({
                    f"gradients/{name}/mean": grad.mean().item(),
                    f"gradients/{name}/std": grad.std().item(),
                    f"gradients/{name}/norm": grad.norm().item()
                })
                
        self.log_metrics({
            "step": step,
            **gradient_stats
        })
        
    def log_robot_trajectories(
        self,
        trajectories: Dict[str, Dict[str, np.ndarray]],
        step: int
    ):
        """
        Log robot trajectories for visualization.
        
        Args:
            trajectories: Dictionary of robot trajectories
            step: Training step
        """
        # Convert trajectories to format suitable for plotting
        plot_data = []
        for robot_name, traj in trajectories.items():
            positions = traj["positions"]
            for t in range(len(positions)):
                plot_data.append([
                    step,
                    robot_name,
                    t,
                    positions[t][0],  # x
                    positions[t][1],  # y
                    positions[t][2]   # z
                ])
                
        # Create trajectory plot
        trajectory_table = wandb.Table(
            data=plot_data,
            columns=["step", "robot", "timestep", "x", "y", "z"]
        )
        
        wandb.log({
            "trajectories": wandb.plot.line(
                trajectory_table,
                "timestep",
                "z",
                title="Robot Trajectories",
                groupby=["robot"]
            )
        })
        
    def log_video(
        self,
        video_frames: List[np.ndarray],
        fps: int,
        step: int
    ):
        """
        Log video of robot behavior.
        
        Args:
            video_frames: List of video frames (numpy arrays)
            fps: Frames per second
            step: Training step
        """
        # Convert frames to uint8 if needed
        if video_frames[0].dtype != np.uint8:
            video_frames = [
                (frame * 255).astype(np.uint8)
                for frame in video_frames
            ]
            
        wandb.log({
            "video": wandb.Video(
                np.stack(video_frames),
                fps=fps,
                format="gif"
            ),
            "step": step
        })
        
    def log_hyperparameters(self, hyperparams: Dict):
        """
        Log hyperparameters for the experiment.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.run.config.update(hyperparams)
        
    def log_model_architecture(self, model):
        """
        Log model architecture summary.
        
        Args:
            model: PyTorch model
        """
        wandb.watch(
            model,
            log="all",
            log_freq=100
        )
        
    def log_environment_info(self, env_info: Dict):
        """
        Log environment information.
        
        Args:
            env_info: Dictionary of environment information
        """
        # Save environment info as JSON artifact
        env_artifact = wandb.Artifact(
            name=f"environment_info_{self.run.id}",
            type="environment"
        )
        
        with env_artifact.new_file("env_info.json") as f:
            json.dump(env_info, f)
            
        self.run.log_artifact(env_artifact)
        
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        checkpoint_name: str
    ):
        """
        Save a checkpoint as a WandB artifact.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            checkpoint_name: Name for the checkpoint artifact
        """
        checkpoint_artifact = wandb.Artifact(
            name=checkpoint_name,
            type="model-checkpoint"
        )
        
        checkpoint_artifact.add_dir(checkpoint_dir)
        self.run.log_artifact(checkpoint_artifact)
        
    def close(self):
        """Clean up WandB run."""
        if self.run is not None:
            self.run.finish()
