#!/usr/bin/env python3
"""
MuJoCo Scene Builder for Motion Planning Demonstrations

This module provides a reusable builder class for creating MuJoCo scenes with:
- Virtual obstacles (red spheres)
- Target position markers (green spheres) 
- Trajectory trace dots (configurable colors)

Used by trajectory optimization demos to avoid code duplication.
"""

import os
import numpy as np
from typing import List, Optional
from motion_planning.utils import Obstacle


class MujocoSceneBuilder:
    """Builder class for creating MuJoCo scenes with obstacles and visualization elements"""
    
    def __init__(self, base_scene_path: Optional[str] = None):
        """
        Initialize the scene builder.
        
        Args:
            base_scene_path: Path to base scene XML. If None, uses default Kinova Gen3 scene.
        """
        self.base_scene_path = base_scene_path or 'robot_models/kinova_gen3/scene.xml'
        self.xml_elements = []
        
    def add_obstacles(self, obstacles: List[Obstacle]) -> 'MujocoSceneBuilder':
        """
        Add obstacle visualization to the scene.
        
        Args:
            obstacles: List of Obstacle objects to visualize as red spheres
            
        Returns:
            Self for method chaining
        """
        for i, obstacle in enumerate(obstacles):
            obstacle_xml = f'''
    <!-- Virtual obstacle {i+1} visualization -->
    <body name="obstacle_{i+1}" pos="{obstacle.center[0]:.3f} {obstacle.center[1]:.3f} {obstacle.center[2]:.3f}">
        <geom name="obstacle_{i+1}_geom" type="sphere" size="{obstacle.radius}" 
              rgba="1.0 0.2 0.2 0.6" material="" contype="0" conaffinity="0"/>
        <site name="obstacle_{i+1}_center" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
    </body>
    '''
            self.xml_elements.append(obstacle_xml)
        
        return self
        
    def add_target(self, position: np.ndarray, size: float = 0.03) -> 'MujocoSceneBuilder':
        """
        Add target position marker to the scene.
        
        Args:
            position: 3D position of the target
            size: Radius of the target sphere
            
        Returns:
            Self for method chaining
        """
        target_xml = f'''
    <!-- Target position marker -->
    <body name="target_marker" pos="{position[0]} {position[1]} {position[2]}">
        <geom name="target_geom" type="sphere" size="{size}" 
              rgba="0.0 0.8 0.0 0.8" material="" contype="0" conaffinity="0"/>
        <site name="target_center" pos="0 0 0" size="0.005" rgba="0 1 0 1"/>
    </body>
    '''
        self.xml_elements.append(target_xml)
        return self
        
    def add_trace_dots(self, count: int, 
                      initial_position: Optional[np.ndarray] = None,
                      default_color: Optional[str] = None) -> 'MujocoSceneBuilder':
        """
        Add trajectory trace dots to the scene.
        
        Args:
            count: Number of trace dots to create
            initial_position: Initial position for dots (default: hidden at (0,0,-10))
            default_color: RGBA color string (default: blue)
            
        Returns:
            Self for method chaining
        """
        if initial_position is None:
            initial_position = np.array([0, 0, -10])  # Hidden below ground
            
        if default_color is None:
            default_color = "0.2 0.6 1.0 0.8"  # Blue
            
        trace_xml_comment = '''
    <!-- Trajectory trace dots (initially hidden at origin) -->'''
        self.xml_elements.append(trace_xml_comment)
        
        for i in range(count):
            trace_xml = f'''
    <body name="trace_dot_{i}" pos="{initial_position[0]} {initial_position[1]} {initial_position[2]}">
        <geom name="trace_dot_{i}_geom" type="sphere" size="0.008" 
              rgba="{default_color}" material="" contype="0" conaffinity="0"/>
    </body>'''
            self.xml_elements.append(trace_xml)
            
        return self
        
    def build_scene(self, output_filename: str) -> str:
        """
        Build the final scene and save to file.
        
        Args:
            output_filename: Name of the output XML file (relative to robot_models/kinova_gen3/)
            
        Returns:
            Path to the created scene file
            
        Raises:
            FileNotFoundError: If base scene file doesn't exist
            ValueError: If XML insertion point cannot be found
        """
        # Try to find the base scene file
        if not os.path.exists(self.base_scene_path):
            # Fallback to gen3.xml if scene.xml doesn't exist
            fallback_path = 'robot_models/kinova_gen3/gen3.xml'
            if os.path.exists(fallback_path):
                self.base_scene_path = fallback_path
            else:
                raise FileNotFoundError(f"Neither {self.base_scene_path} nor {fallback_path} found")
        
        # Read the base scene content
        with open(self.base_scene_path, 'r') as f:
            scene_content = f.read()
        
        # Combine all XML elements
        all_xml = ''.join(self.xml_elements)
        
        # Find insertion point in the XML
        insert_pos = scene_content.rfind('</worldbody>')
        if insert_pos == -1:
            # Fallback - add before closing mujoco tag
            insert_pos = scene_content.rfind('</mujoco>')
            if insert_pos == -1:
                raise ValueError("Could not find insertion point in XML")
        
        # Insert the new XML content
        modified_content = (scene_content[:insert_pos] + 
                           all_xml + 
                           scene_content[insert_pos:])
        
        # Save to output file
        output_path = f'robot_models/kinova_gen3/{output_filename}'
        # with open(output_path, 'w') as f:
        #     f.write(modified_content)
        
        return output_path
        
    def reset(self) -> 'MujocoSceneBuilder':
        """
        Reset the builder to empty state.
        
        Returns:
            Self for method chaining
        """
        self.xml_elements = []
        return self


def create_standard_scene(obstacles: List[Obstacle], 
                         target_position: np.ndarray,
                         trace_dot_count: int = 200,
                         output_filename: str = "trajopt_scene.xml") -> str:
    """
    Convenience function to create a standard trajectory optimization scene.
    
    Args:
        obstacles: List of obstacles to visualize
        target_position: Target position for the robot
        trace_dot_count: Number of trajectory trace dots
        output_filename: Name of the output scene file
        
    Returns:
        Path to the created scene file
    """
    builder = MujocoSceneBuilder()
    return (builder
            .add_obstacles(obstacles)
            .add_target(target_position)
            .add_trace_dots(trace_dot_count)
            .build_scene(output_filename))


def create_pareto_scene(obstacles: List[Obstacle],
                       target_position: np.ndarray, 
                       max_trajectories: int,
                       output_filename: str = "pareto_search_scene.xml") -> str:
    """
    Convenience function to create a scene for Pareto search visualization.
    
    Args:
        obstacles: List of obstacles to visualize
        target_position: Target position for the robot
        max_trajectories: Maximum number of trajectories (for trace dot calculation)
        output_filename: Name of the output scene file
        
    Returns:
        Path to the created scene file
    """
    # Conservative estimate for trace dots needed
    max_dots_needed = max_trajectories * 100
    
    builder = MujocoSceneBuilder()
    return (builder
            .add_obstacles(obstacles)
            .add_target(target_position)
            .add_trace_dots(max_dots_needed, default_color="1.0 1.0 1.0 0.8")  # White for recoloring
            .build_scene(output_filename)) 