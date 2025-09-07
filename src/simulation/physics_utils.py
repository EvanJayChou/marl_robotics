"""
Utilities for physics calculations and transformations in the simulation environment.
"""

import numpy as np
from typing import List, Tuple, Union
import transforms3d as tf3d
import torch

class PhysicsUtils:
    @staticmethod
    def quaternion_to_euler(quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert quaternion to euler angles (roll, pitch, yaw).
        
        Args:
            quaternion: [w, x, y, z] quaternion
            
        Returns:
            np.ndarray: [roll, pitch, yaw] in radians
        """
        return tf3d.euler.quat2euler(quaternion, 'sxyz')
        
    @staticmethod
    def euler_to_quaternion(euler: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert euler angles to quaternion.
        
        Args:
            euler: [roll, pitch, yaw] in radians
            
        Returns:
            np.ndarray: [w, x, y, z] quaternion
        """
        return tf3d.euler.euler2quat(*euler, 'sxyz')
        
    @staticmethod
    def compute_com_velocity(
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Compute center of mass velocity for a multi-body system.
        
        Args:
            positions: (N, 3) array of positions
            velocities: (N, 3) array of velocities
            masses: (N,) array of masses
            
        Returns:
            np.ndarray: (3,) COM velocity
        """
        total_mass = np.sum(masses)
        com_vel = np.sum(velocities * masses[:, None], axis=0) / total_mass
        return com_vel
        
    @staticmethod
    def compute_angular_momentum(
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        com_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute angular momentum about the center of mass.
        
        Args:
            positions: (N, 3) array of positions
            velocities: (N, 3) array of velocities
            masses: (N,) array of masses
            com_position: (3,) center of mass position
            
        Returns:
            np.ndarray: (3,) angular momentum vector
        """
        rel_positions = positions - com_position
        angular_momentum = np.zeros(3)
        
        for i in range(len(masses)):
            r = rel_positions[i]
            v = velocities[i]
            m = masses[i]
            angular_momentum += m * np.cross(r, v)
            
        return angular_momentum
        
    @staticmethod
    def compute_zmp(
        com_position: np.ndarray,
        com_velocity: np.ndarray,
        com_acceleration: np.ndarray,
        gravity: float = 9.81
    ) -> np.ndarray:
        """
        Compute Zero Moment Point (ZMP) for balance control.
        
        Args:
            com_position: (3,) COM position
            com_velocity: (3,) COM velocity
            com_acceleration: (3,) COM acceleration
            gravity: gravitational acceleration
            
        Returns:
            np.ndarray: (2,) ZMP position (x, y)
        """
        z_h = com_position[2]
        
        zmp_x = com_position[0] - (z_h / gravity) * com_acceleration[0]
        zmp_y = com_position[1] - (z_h / gravity) * com_acceleration[1]
        
        return np.array([zmp_x, zmp_y])
        
    @staticmethod
    def compute_support_polygon(
        contact_points: np.ndarray,
        contact_normals: np.ndarray
    ) -> np.ndarray:
        """
        Compute the support polygon from contact points.
        
        Args:
            contact_points: (N, 3) contact point positions
            contact_normals: (N, 3) contact point normals
            
        Returns:
            np.ndarray: (M, 2) vertices of the support polygon
        """
        # Project points onto ground plane (xy-plane)
        points_2d = contact_points[:, :2]
        
        # Compute convex hull of projected points
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_2d)
        
        return points_2d[hull.vertices]
        
    @staticmethod
    def check_static_stability(
        zmp: np.ndarray,
        support_polygon: np.ndarray
    ) -> bool:
        """
        Check if ZMP lies within the support polygon.
        
        Args:
            zmp: (2,) ZMP position
            support_polygon: (N, 2) vertices of support polygon
            
        Returns:
            bool: True if statically stable
        """
        from scipy.spatial import Delaunay
        hull = Delaunay(support_polygon)
        return hull.find_simplex(zmp) >= 0
        
    @staticmethod
    def compute_joint_torques(
        jacobian: np.ndarray,
        force: np.ndarray
    ) -> np.ndarray:
        """
        Compute joint torques from end-effector force using the Jacobian transpose.
        
        Args:
            jacobian: (6, N) Jacobian matrix
            force: (6,) force/torque vector
            
        Returns:
            np.ndarray: (N,) joint torques
        """
        return jacobian.T @ force
