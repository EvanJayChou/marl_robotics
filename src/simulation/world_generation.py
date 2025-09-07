"""
World generation utilities for creating diverse training environments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom

class WorldGeneration:
    def __init__(self, config: dict):
        """
        Initialize world generation with configuration parameters.
        
        Args:
            config (dict): Configuration for world generation
        """
        self.config = config
        self.terrain_types = {
            "flat": self._generate_flat_terrain,
            "rough": self._generate_rough_terrain,
            "stairs": self._generate_stairs_terrain,
            "gaps": self._generate_gaps_terrain,
            "slopes": self._generate_slopes_terrain
        }
        
    def generate_terrain(
        self,
        terrain_type: str,
        size: Tuple[float, float],
        parameters: Dict
    ) -> str:
        """
        Generate terrain of specified type.
        
        Args:
            terrain_type: Type of terrain to generate
            size: (length, width) of terrain
            parameters: Additional parameters for terrain generation
            
        Returns:
            str: Prim path of generated terrain
        """
        if terrain_type not in self.terrain_types:
            raise ValueError(f"Unknown terrain type: {terrain_type}")
            
        return self.terrain_types[terrain_type](size, parameters)
        
    def _generate_flat_terrain(
        self,
        size: Tuple[float, float],
        parameters: Dict
    ) -> str:
        """Generate flat terrain."""
        length, width = size
        prim_path = "/World/Terrain/Flat"
        
        ground = prim_utils.create_prim(
            prim_path,
            "Plane",
            position=[0, 0, 0],
            scale=[length/2, width/2, 1]
        )
        
        return prim_path
        
    def _generate_rough_terrain(
        self,
        size: Tuple[float, float],
        parameters: Dict
    ) -> str:
        """Generate rough terrain using heightfield."""
        length, width = size
        prim_path = "/World/Terrain/Rough"
        
        # Generate height field
        resolution = parameters.get("resolution", 128)
        amplitude = parameters.get("amplitude", 0.2)
        frequency = parameters.get("frequency", 0.1)
        
        x = np.linspace(0, length, resolution)
        y = np.linspace(0, width, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Generate Perlin noise
        from noise import pnoise2
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                Z[i,j] = pnoise2(X[i,j]*frequency, 
                                Y[i,j]*frequency, 
                                octaves=4) * amplitude
                                
        # Create height field
        terrain = UsdGeom.Mesh.Define(stage, prim_path)
        points = []
        for i in range(resolution):
            for j in range(resolution):
                points.append(Gf.Vec3f(X[i,j], Y[i,j], Z[i,j]))
                
        terrain.CreatePointsAttr(points)
        
        return prim_path
        
    def _generate_stairs_terrain(
        self,
        size: Tuple[float, float],
        parameters: Dict
    ) -> str:
        """Generate terrain with stairs."""
        length, width = size
        prim_path = "/World/Terrain/Stairs"
        
        step_height = parameters.get("step_height", 0.2)
        step_depth = parameters.get("step_depth", 0.4)
        num_steps = int(length / step_depth)
        
        for i in range(num_steps):
            step = prim_utils.create_prim(
                f"{prim_path}/Step_{i}",
                "Cube",
                position=[i * step_depth, 0, i * step_height / 2],
                scale=[step_depth/2, width/2, step_height * (i+1)/2]
            )
            
        return prim_path
        
    def _generate_gaps_terrain(
        self,
        size: Tuple[float, float],
        parameters: Dict
    ) -> str:
        """Generate terrain with gaps."""
        length, width = size
        prim_path = "/World/Terrain/Gaps"
        
        gap_width = parameters.get("gap_width", 0.5)
        platform_length = parameters.get("platform_length", 2.0)
        num_gaps = int(length / (gap_width + platform_length))
        
        for i in range(num_gaps + 1):
            platform = prim_utils.create_prim(
                f"{prim_path}/Platform_{i}",
                "Cube",
                position=[i * (gap_width + platform_length), 0, 0],
                scale=[platform_length/2, width/2, 0.1]
            )
            
        return prim_path
        
    def _generate_slopes_terrain(
        self,
        size: Tuple[float, float],
        parameters: Dict
    ) -> str:
        """Generate terrain with varying slopes."""
        length, width = size
        prim_path = "/World/Terrain/Slopes"
        
        max_slope = parameters.get("max_slope", 15)  # degrees
        segment_length = parameters.get("segment_length", 5.0)
        num_segments = int(length / segment_length)
        
        current_height = 0
        for i in range(num_segments):
            slope_angle = np.random.uniform(-max_slope, max_slope)
            height_change = segment_length * np.tan(np.radians(slope_angle))
            
            slope = prim_utils.create_prim(
                f"{prim_path}/Slope_{i}",
                "Cube",
                position=[i * segment_length, 0, current_height],
                rotation=[0, -slope_angle, 0],
                scale=[segment_length/2, width/2, 0.1]
            )
            
            current_height += height_change
            
        return prim_path
        
    def add_obstacles(
        self,
        terrain_path: str,
        num_obstacles: int,
        parameters: Dict
    ) -> List[str]:
        """
        Add random obstacles to the terrain.
        
        Args:
            terrain_path: Path to terrain prim
            num_obstacles: Number of obstacles to add
            parameters: Obstacle parameters
            
        Returns:
            List[str]: Paths to created obstacles
        """
        obstacle_paths = []
        min_spacing = parameters.get("min_spacing", 2.0)
        
        # Get terrain bounds
        terrain_prim = prim_utils.get_prim_at_path(terrain_path)
        bounds = prim_utils.get_prim_bounds(terrain_prim)
        
        # Generate obstacle positions
        positions = self._generate_obstacle_positions(
            num_obstacles,
            bounds,
            min_spacing
        )
        
        # Create obstacles
        for i, pos in enumerate(positions):
            obstacle_type = np.random.choice(["cube", "sphere", "cylinder"])
            obstacle_path = self._create_obstacle(
                f"{terrain_path}/Obstacle_{i}",
                obstacle_type,
                pos,
                parameters
            )
            obstacle_paths.append(obstacle_path)
            
        return obstacle_paths
        
    def _generate_obstacle_positions(
        self,
        num_obstacles: int,
        bounds: List[float],
        min_spacing: float
    ) -> List[List[float]]:
        """Generate valid obstacle positions."""
        positions = []
        attempts = 0
        max_attempts = 1000
        
        while len(positions) < num_obstacles and attempts < max_attempts:
            pos = [
                np.random.uniform(bounds[0], bounds[1]),
                np.random.uniform(bounds[2], bounds[3]),
                0  # Height will be adjusted based on terrain
            ]
            
            # Check spacing with existing obstacles
            valid = True
            for existing_pos in positions:
                if np.linalg.norm(np.array(pos) - np.array(existing_pos)) < min_spacing:
                    valid = False
                    break
                    
            if valid:
                positions.append(pos)
                
            attempts += 1
            
        return positions
        
    def _create_obstacle(
        self,
        prim_path: str,
        obstacle_type: str,
        position: List[float],
        parameters: Dict
    ) -> str:
        """Create an obstacle of specified type."""
        size = parameters.get("obstacle_size", 0.5)
        
        if obstacle_type == "cube":
            prim = prim_utils.create_prim(
                prim_path,
                "Cube",
                position=position,
                scale=[size/2, size/2, size/2]
            )
        elif obstacle_type == "sphere":
            prim = prim_utils.create_prim(
                prim_path,
                "Sphere",
                position=position,
                scale=[size/2, size/2, size/2]
            )
        elif obstacle_type == "cylinder":
            prim = prim_utils.create_prim(
                prim_path,
                "Cylinder",
                position=position,
                scale=[size/2, size/2, size]
            )
            
        return prim_path
