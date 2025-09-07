"""
Sensor simulation for bipedal robots, including IMU, force/torque sensors, and cameras.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

class SensorSimulation:
    def __init__(self, config: dict):
        """
        Initialize sensor simulation with noise parameters.
        
        Args:
            config (dict): Configuration for sensor parameters and noise
        """
        self.config = config
        self._setup_noise_parameters()
        
    def _setup_noise_parameters(self):
        """Setup sensor noise parameters from config."""
        self.imu_noise = {
            "accel": self.config.get("imu_accel_noise", 0.01),
            "gyro": self.config.get("imu_gyro_noise", 0.001),
            "bias_accel": self.config.get("imu_accel_bias", 0.05),
            "bias_gyro": self.config.get("imu_gyro_bias", 0.01)
        }
        
        self.ft_noise = {
            "force": self.config.get("ft_force_noise", 0.1),
            "torque": self.config.get("ft_torque_noise", 0.01)
        }
        
        self.encoder_noise = self.config.get("encoder_noise", 0.001)
        
    def simulate_imu(
        self,
        linear_acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        orientation: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Simulate IMU measurements with noise and bias.
        
        Args:
            linear_acceleration: (3,) True linear acceleration
            angular_velocity: (3,) True angular velocity
            orientation: (4,) True orientation quaternion
            
        Returns:
            Dict containing noisy IMU measurements
        """
        # Add noise and bias to accelerometer
        accel_noise = np.random.normal(0, self.imu_noise["accel"], 3)
        accel_bias = np.random.normal(0, self.imu_noise["bias_accel"], 3)
        noisy_accel = linear_acceleration + accel_noise + accel_bias
        
        # Add noise and bias to gyroscope
        gyro_noise = np.random.normal(0, self.imu_noise["gyro"], 3)
        gyro_bias = np.random.normal(0, self.imu_noise["bias_gyro"], 3)
        noisy_gyro = angular_velocity + gyro_noise + gyro_bias
        
        return {
            "linear_acceleration": noisy_accel,
            "angular_velocity": noisy_gyro,
            "orientation": orientation  # Quaternion
        }
        
    def simulate_force_torque_sensor(
        self,
        force: np.ndarray,
        torque: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Simulate force/torque sensor measurements.
        
        Args:
            force: (3,) True force vector
            torque: (3,) True torque vector
            
        Returns:
            Dict containing noisy force/torque measurements
        """
        # Add noise to force measurements
        force_noise = np.random.normal(0, self.ft_noise["force"], 3)
        noisy_force = force + force_noise
        
        # Add noise to torque measurements
        torque_noise = np.random.normal(0, self.ft_noise["torque"], 3)
        noisy_torque = torque + torque_noise
        
        return {
            "force": noisy_force,
            "torque": noisy_torque
        }
        
    def simulate_joint_encoders(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Simulate joint encoder measurements.
        
        Args:
            joint_positions: (N,) True joint positions
            joint_velocities: (N,) True joint velocities
            
        Returns:
            Dict containing noisy joint measurements
        """
        # Add noise to position measurements
        pos_noise = np.random.normal(0, self.encoder_noise, joint_positions.shape)
        noisy_positions = joint_positions + pos_noise
        
        # Add noise to velocity measurements (derived from position)
        vel_noise = np.random.normal(0, self.encoder_noise * 10, joint_velocities.shape)
        noisy_velocities = joint_velocities + vel_noise
        
        return {
            "positions": noisy_positions,
            "velocities": noisy_velocities
        }
        
    def simulate_contact_sensors(
        self,
        contact_states: np.ndarray,
        contact_forces: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Simulate binary contact sensors and force sensors at the feet.
        
        Args:
            contact_states: (N,) Binary array of true contact states
            contact_forces: (N, 3) Array of contact forces
            
        Returns:
            Dict containing contact sensor measurements
        """
        # Add noise to contact force measurements
        force_noise = np.random.normal(0, self.ft_noise["force"], contact_forces.shape)
        noisy_forces = contact_forces + force_noise
        
        # Threshold forces to determine contact state
        force_magnitude = np.linalg.norm(noisy_forces, axis=1)
        contact_threshold = self.config.get("contact_threshold", 5.0)
        detected_contacts = force_magnitude > contact_threshold
        
        return {
            "contact_state": detected_contacts,
            "contact_forces": noisy_forces
        }
        
    def simulate_depth_camera(
        self,
        depth_image: np.ndarray,
        camera_params: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Simulate depth camera measurements with noise.
        
        Args:
            depth_image: (H, W) True depth values
            camera_params: Camera parameters (FOV, resolution, etc.)
            
        Returns:
            Dict containing processed depth image and point cloud
        """
        # Add multiplicative noise to depth measurements
        depth_noise_factor = 1.0 + np.random.normal(
            0,
            self.config.get("depth_noise", 0.01),
            depth_image.shape
        )
        noisy_depth = depth_image * depth_noise_factor
        
        # Convert depth image to point cloud
        point_cloud = self._depth_to_pointcloud(noisy_depth, camera_params)
        
        return {
            "depth_image": noisy_depth,
            "point_cloud": point_cloud
        }
        
    def _depth_to_pointcloud(
        self,
        depth_image: np.ndarray,
        camera_params: Dict
    ) -> np.ndarray:
        """Convert depth image to point cloud."""
        height, width = depth_image.shape
        fx = camera_params["focal_length_x"]
        fy = camera_params["focal_length_y"]
        cx = camera_params["center_x"]
        cy = camera_params["center_y"]
        
        # Create pixel coordinate grid
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D points
        z = depth_image
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy
        
        # Stack points into (N, 3) array
        points = np.stack([x, y, z], axis=-1)
        
        # Reshape to (N, 3)
        return points.reshape(-1, 3)
