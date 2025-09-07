"""
Interface for Isaac Gym simulation environment, specialized for multi-agent robotics.
"""

import isaacgym
import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *

class IsaacGymInterface:
    def __init__(self, config: dict):
        """
        Initialize Isaac Gym interface with configuration parameters.
        
        Args:
            config (dict): Configuration for simulation parameters
        """
        self.config = config
        self.gym = gymapi.acquire_gym()
        
        # Set up simulation parameters
        self._setup_sim()
        
        # Robot and environment handles
        self.envs = []
        self.robot_handles = {}
        self.dof_states = None
        self.root_states = None
        
    def _setup_sim(self):
        """Initialize simulation parameters and create sim."""
        # Configure sim parameters
        self.sim_params = gymapi.SimParams()
        
        # Physics parameters
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # Set physics engine parameters
        self.sim_params.physx.solver_type = 1  # TGS
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.contact_offset = 0.02
        self.sim_params.physx.rest_offset = 0.0
        
        # Create sim
        self.sim = self.gym.create_sim(
            compute_device=0,
            graphics_device=0,
            type=gymapi.SIM_PHYSX,
            params=self.sim_params
        )
        
        if self.sim is None:
            raise Exception("Failed to create sim")
            
    def create_ground_plane(self):
        """Create ground plane in simulation."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0
        
        self.gym.add_ground(self.sim, plane_params)
        
    def create_environments(
        self,
        num_envs: int,
        spacing: float
    ):
        """
        Create multiple simulation environments.
        
        Args:
            num_envs: Number of parallel environments
            spacing: Spacing between environments
        """
        # Create environment array
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(num_envs):
            env = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(num_envs))
            )
            self.envs.append(env)
            
    def load_robot(
        self,
        robot_file: str,
        num_envs: int
    ):
        """
        Load robot asset and create actors in environments.
        
        Args:
            robot_file: Path to robot URDF/USD file
            num_envs: Number of environments to spawn robots in
        """
        # Load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.armature = 0.01
        asset_options.disable_gravity = False
        
        robot_asset = self.gym.load_asset(
            self.sim,
            "",  # Asset root
            robot_file,
            asset_options
        )
        
        # Create robot actors in each environment
        self.robot_handles = {}
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # Starting height
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        for i, env in enumerate(self.envs):
            robot_handle = self.gym.create_actor(
                env,
                robot_asset,
                pose,
                f"robot_{i}",
                i,
                1  # Body group
            )
            
            self.robot_handles[i] = robot_handle
            
            # Set default joint positions
            props = self.gym.get_actor_dof_properties(env, robot_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)
            props["damping"].fill(100.0)
            self.gym.set_actor_dof_properties(env, robot_handle, props)
            
        # Get state tensors
        self._get_state_tensors()
        
    def _get_state_tensors(self):
        """Get state tensors for efficient access to simulation state."""
        # Get state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_states = gymtorch.wrap_tensor(dof_state)
        
    def step_simulation(self):
        """Step physics simulation forward."""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
    def get_robot_state(self, env_idx: int) -> Dict:
        """
        Get current state of robot in specified environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Dict containing robot state information
        """
        robot_handle = self.robot_handles[env_idx]
        
        # Get DOF states
        dof_indices = self.gym.get_actor_dof_indices(
            self.envs[env_idx],
            robot_handle,
            gymapi.STATE_ALL
        )
        
        dof_states = {
            "position": self.dof_states[dof_indices, 0].cpu().numpy(),
            "velocity": self.dof_states[dof_indices, 1].cpu().numpy()
        }
        
        # Get root state
        actor_idx = self.gym.get_actor_index(
            self.envs[env_idx],
            robot_handle,
            gymapi.DOMAIN_SIM
        )
        
        root_state = {
            "position": self.root_states[actor_idx, 0:3].cpu().numpy(),
            "orientation": self.root_states[actor_idx, 3:7].cpu().numpy(),
            "linear_vel": self.root_states[actor_idx, 7:10].cpu().numpy(),
            "angular_vel": self.root_states[actor_idx, 10:13].cpu().numpy()
        }
        
        return {
            "dof_states": dof_states,
            "root_state": root_state
        }
        
    def set_robot_state(
        self,
        env_idx: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None
    ):
        """
        Set robot joint positions and velocities.
        
        Args:
            env_idx: Environment index
            positions: Joint positions
            velocities: Joint velocities (optional)
        """
        robot_handle = self.robot_handles[env_idx]
        
        if velocities is None:
            velocities = np.zeros_like(positions)
            
        # Set DOF states
        self.gym.set_actor_dof_states(
            self.envs[env_idx],
            robot_handle,
            gymapi.STATE_ALL,
            positions.tolist()
        )
        
    def apply_robot_actions(
        self,
        env_idx: int,
        actions: np.ndarray
    ):
        """
        Apply actions to robot.
        
        Args:
            env_idx: Environment index
            actions: Joint position or torque commands
        """
        robot_handle = self.robot_handles[env_idx]
        
        # Apply actions based on control mode
        if self.config.get("control_mode") == "position":
            self.gym.set_actor_dof_position_targets(
                self.envs[env_idx],
                robot_handle,
                actions.tolist()
            )
        else:  # torque control
            self.gym.apply_actor_dof_efforts(
                self.envs[env_idx],
                robot_handle,
                actions.tolist()
            )
            
    def get_camera_image(
        self,
        env_idx: int,
        camera_handle: int
    ) -> Dict[str, np.ndarray]:
        """
        Get RGB and depth images from camera.
        
        Args:
            env_idx: Environment index
            camera_handle: Handle to camera sensor
            
        Returns:
            Dict containing RGB and depth images
        """
        # Get camera image
        self.gym.render_all_camera_sensors(self.sim)
        
        rgb = self.gym.get_camera_image(
            self.sim,
            self.envs[env_idx],
            camera_handle,
            gymapi.IMAGE_COLOR
        )
        
        depth = self.gym.get_camera_image(
            self.sim,
            self.envs[env_idx],
            camera_handle,
            gymapi.IMAGE_DEPTH
        )
        
        return {
            "rgb": rgb,
            "depth": depth
        }
        
    def close(self):
        """Clean up simulation resources."""
        self.gym.destroy_sim(self.sim)
