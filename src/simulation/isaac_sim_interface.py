"""
Interface for Isaac Sim simulation environment for multi-agent bipedal locomotion.
Provides high-level abstractions for physics simulation and robot control.
"""

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.world import World
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

class IsaacSimInterface:
    def __init__(self, config: dict):
        """
        Initialize the Isaac Sim interface with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.world = None
        self.robots = {}
        self.terrain = None
        self._physics_dt = 1.0 / 60.0
        self._rendering_dt = 1.0 / 60.0
        
        # Initialize the simulation world
        self._initialize_world()
        
    def _initialize_world(self):
        """Initialize the Isaac Sim world with physics settings."""
        self.world = World(physics_dt=self._physics_dt, 
                         rendering_dt=self._rendering_dt,
                         stage_units_in_meters=1.0)
        
        # Set up physics parameters
        self.world.set_physics_params(
            solver_type=self.config.get("solver_type", "TGS"),
            num_substeps=self.config.get("num_substeps", 2),
            enable_gravity=True
        )
        
    def spawn_robot(self, robot_name: str, position: List[float], 
                   orientation: List[float]) -> str:
        """
        Spawn a bipedal robot in the simulation.
        
        Args:
            robot_name (str): Unique identifier for the robot
            position (List[float]): [x, y, z] spawn position
            orientation (List[float]): [roll, pitch, yaw] orientation
            
        Returns:
            str: Unique identifier for the spawned robot
        """
        # Load robot URDF/USD
        robot_path = self.config["robot_models"][robot_name]
        robot_prim = prim_utils.create_prim(
            f"/World/Robots/{robot_name}",
            usd_path=robot_path,
            position=position,
            orientation=orientation
        )
        
        # Create articulation view for control
        self.robots[robot_name] = ArticulationView(
            prim_paths_expr=f"/World/Robots/{robot_name}"
        )
        
        return robot_name
        
    def generate_terrain(self, terrain_type: str, parameters: Dict):
        """
        Generate terrain for the simulation environment.
        
        Args:
            terrain_type (str): Type of terrain to generate
            parameters (Dict): Terrain generation parameters
        """
        # Implementation for different terrain types
        if terrain_type == "flat":
            self._generate_flat_terrain(parameters)
        elif terrain_type == "rough":
            self._generate_rough_terrain(parameters)
        elif terrain_type == "stairs":
            self._generate_stairs_terrain(parameters)
            
    def _generate_flat_terrain(self, parameters: Dict):
        """Generate flat terrain with given parameters."""
        ground_prim = prim_utils.create_prim(
            "/World/Ground",
            "Plane",
            position=[0, 0, 0],
            scale=[parameters.get("size", 100), 
                   parameters.get("size", 100), 1]
        )
        
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Step the simulation forward with given actions.
        
        Args:
            actions (Dict[str, np.ndarray]): Actions for each robot
            
        Returns:
            Tuple containing:
            - observations (Dict): Current observations for each robot
            - rewards (Dict): Rewards for each robot
            - dones (Dict): Done flags for each robot
            - info (Dict): Additional information
        """
        # Apply actions to robots
        for robot_name, action in actions.items():
            if robot_name in self.robots:
                self.robots[robot_name].apply_action(action)
                
        # Step physics simulation
        self.world.step(render=True)
        
        # Collect observations and rewards
        observations = self._get_observations()
        rewards = self._compute_rewards()
        dones = self._check_termination()
        info = self._get_info()
        
        return observations, rewards, dones, info
        
    def _get_observations(self) -> Dict:
        """Collect observations from all robots."""
        observations = {}
        for robot_name, robot in self.robots.items():
            observations[robot_name] = {
                "joint_positions": robot.get_joint_positions(),
                "joint_velocities": robot.get_joint_velocities(),
                "base_position": robot.get_world_pose()[0],
                "base_orientation": robot.get_world_pose()[1]
            }
        return observations
        
    def _compute_rewards(self) -> Dict:
        """Compute rewards for all robots."""
        rewards = {}
        for robot_name, robot in self.robots.items():
            # Implement reward computation based on task objectives
            rewards[robot_name] = self._compute_single_robot_reward(robot)
        return rewards
        
    def _compute_single_robot_reward(self, robot: ArticulationView) -> float:
        """Compute reward for a single robot based on task objectives."""
        # Example reward computation
        forward_progress = robot.get_world_pose()[0][0]  # X-position
        energy_penalty = np.sum(np.abs(robot.get_joint_velocities()))
        
        return forward_progress - 0.1 * energy_penalty
        
    def _check_termination(self) -> Dict:
        """Check termination conditions for all robots."""
        dones = {}
        for robot_name, robot in self.robots.items():
            # Check termination conditions (e.g., falling, reaching goal)
            base_height = robot.get_world_pose()[0][2]
            dones[robot_name] = base_height < 0.3  # Example: robot has fallen
        return dones
        
    def _get_info(self) -> Dict:
        """Get additional information about the simulation state."""
        info = {}
        for robot_name, robot in self.robots.items():
            info[robot_name] = {
                "base_velocity": robot.get_linear_velocity(),
                "base_angular_velocity": robot.get_angular_velocity()
            }
        return info
        
    def reset(self) -> Dict:
        """Reset the simulation environment."""
        self.world.reset()
        
        # Reset robots to initial positions
        for robot_name, robot in self.robots.items():
            initial_pose = self.config["initial_poses"][robot_name]
            robot.set_world_pose(
                position=initial_pose["position"],
                orientation=initial_pose["orientation"]
            )
            
        return self._get_observations()
        
    def close(self):
        """Clean up simulation resources."""
        if self.world is not None:
            self.world.clear()
            self.world = None
