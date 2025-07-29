#!/usr/bin/env python3
"""
Base Trajectory Optimization Demo Class

This module provides a reusable base class for trajectory optimization demonstrations
that abstracts common functionality like:
- Scene creation and management
- Inverse kinematics solving
- Trajectory planning workflow
- Trajectory analysis and validation
- Viewer setup and execution loops

Child classes override specific methods to customize behavior for different planners
and cost function configurations.
"""

import mujoco
print("##########################################")
print(mujoco.__version__)
import numpy as np
import time
import sys
import os
from typing import List, Optional, Union
from abc import ABC, abstractmethod

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.utils import Obstacle
from scene_builder import create_standard_scene, create_pareto_scene
from trajectory_visualizer import TrajectoryVisualizationManager


class TrajectoryOptimizationDemo(ABC):
    """Base class for trajectory optimization demonstrations"""
    
    def __init__(self):
        """Initialize the demo with obstacle and configuration definitions"""
        self.obstacles = self.define_obstacles()
        self.target_position = self.define_target_position()
        self.start_config = self.define_start_config()
        
    @abstractmethod
    def define_obstacles(self) -> List[Obstacle]:
        """Define the obstacles for this demo. Override in child classes."""
        pass
        
    @abstractmethod
    def define_target_position(self) -> np.ndarray:
        """Define the target position for this demo. Override in child classes."""
        pass
        
    @abstractmethod
    def define_start_config(self) -> np.ndarray:
        """Define the start configuration for this demo. Override in child classes."""
        pass
        
    @abstractmethod
    def create_planner(self, model, data):
        """Create and configure the trajectory planner. Override in child classes."""
        pass
        
    @abstractmethod
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for the planner. Return True if successful. Override in child classes."""
        pass
        
    def get_trace_dot_count(self) -> int:
        """Get the number of trace dots for visualization. Override if needed."""
        return 200
        
    def get_scene_filename(self) -> str:
        """Get the scene filename. Override if needed."""
        return "trajopt_demo_scene.xml"
        
    def get_planning_params(self) -> dict:
        """Get planning parameters. Override if needed."""
        return {
            'n_waypoints': 50,
            'dt': 0.1,
            'tolerance': 0.001,
            'max_iterations': 2000
        }
        
    def create_scene(self) -> str:
        """Create the MuJoCo scene with obstacles and target"""
        return create_standard_scene(
            obstacles=self.obstacles,
            target_position=self.target_position,
            trace_dot_count=self.get_trace_dot_count(),
            output_filename=self.get_scene_filename()
        )
        
    def solve_inverse_kinematics(self, kinematics) -> Optional[np.ndarray]:
        """Solve inverse kinematics for the target position"""
        params = self.get_planning_params()
        goal_config, ik_success = kinematics.inverse_kinematics(
            self.target_position,
            initial_guess=self.start_config,
            tolerance=params['tolerance'],
            max_iterations=params['max_iterations']
        )
        
        if not ik_success:
            print("IK failed - target may be unreachable")
            return None
            
        return goal_config
        
    def analyze_trajectory(self, trajectory: List[np.ndarray], kinematics) -> dict:
        """Analyze the planned trajectory for safety and accuracy"""
        stats = {
            'total_length': 0.0,
            'min_obstacle_distances': [float('inf')] * len(self.obstacles),
            'all_safe': True,
            'final_error': float('inf')
        }
        
        # Calculate trajectory metrics
        for i in range(len(trajectory) - 1):
            joint_distance = np.linalg.norm(trajectory[i+1] - trajectory[i])
            stats['total_length'] += joint_distance
            
            # Check distances to all obstacles
            ee_pos, _ = kinematics.forward_kinematics(trajectory[i])
            for j, obstacle in enumerate(self.obstacles):
                obstacle_distance = np.linalg.norm(ee_pos - obstacle.center) - obstacle.radius
                stats['min_obstacle_distances'][j] = min(
                    stats['min_obstacle_distances'][j], 
                    float(obstacle_distance)
                )
        
        # Check safety for all obstacles
        for j, (obstacle, min_dist) in enumerate(zip(self.obstacles, stats['min_obstacle_distances'])):
            if min_dist < obstacle.safe_distance:
                stats['all_safe'] = False
        
        # Verify final position accuracy
        final_ee_pos, _ = kinematics.forward_kinematics(trajectory[-1])
        stats['final_error'] = np.linalg.norm(final_ee_pos - self.target_position)
        
        return stats
        
    def print_planning_results(self, success: bool, planning_time: float, stats: Optional[dict] = None):
        """Print the results of trajectory planning"""
        if success and stats:
            print(f"TrajOpt planning successful! (Planning time: {planning_time:.2f}s)")
            
            if stats['all_safe']:
                print("Trajectory maintains safe distance from all obstacles")
            else:
                print("Warning: Trajectory too close to one or more obstacles!")
                
            print(f"Final positioning error: {stats['final_error']*1000:.1f}mm")
        else:
            print("TrajOpt planning failed!")
            
    def plan_trajectory(self, model, data, kinematics):
        """Main trajectory planning workflow"""
        # Solve inverse kinematics
        goal_config = self.solve_inverse_kinematics(kinematics)
        if goal_config is None:
            return None
            
        # Create and configure planner
        planner = self.create_planner(model, data)
        if not self.setup_cost_functions(planner, kinematics):
            return None
            
        # Plan trajectory
        print("Planning optimal trajectory...")
        start_time = time.time()
        trajectory, success = planner.plan(self.start_config, goal_config)
        planning_time = time.time() - start_time
        
        # Analyze and report results
        if success:
            stats = self.analyze_trajectory(trajectory, kinematics)
            self.print_planning_results(success, planning_time, stats)
            return trajectory
        else:
            self.print_planning_results(success, planning_time)
            return None
            
    def setup_viewer(self, model, data, viewer_handle):
        """Setup the MuJoCo viewer with initial robot position"""
        data.qpos[:7] = self.start_config
        data.ctrl[:7] = self.start_config
        mujoco.mj_forward(model, data)
        viewer_handle.sync()
        time.sleep(1)
        
    def execute_trajectory(self, viewer_handle, model, data, kinematics, trajectory):
        """Execute a trajectory using the visualization manager"""
        visualizer = TrajectoryVisualizationManager(model, data, viewer_handle, kinematics)
        visualizer.run_interactive_session(
            trajectory=trajectory,
            home_config=self.start_config,
            target_position=self.target_position
        )
        
    def execute_planning_loop(self, model, data, kinematics, viewer_handle):
        """Execute the main planning and visualization loop. Override for custom behavior."""
        trajectory = self.plan_trajectory(model, data, kinematics)
        
        if trajectory is not None:
            self.execute_trajectory(viewer_handle, model, data, kinematics, trajectory)
        else:
            print("Cannot proceed with visualization due to planning failure.")
            print("Press Ctrl+C to exit...")
            try:
                while True:
                    mujoco.mj_step(model, data)
                    viewer_handle.sync()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Exiting demo...")
                
    def run_demo(self):
        """Main demo execution method"""
        print(f"{self.__class__.__name__} Demo")
        
        try:
            # Create scene and load model
            model_path = self.create_scene()
            print(model_path)
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)
            kinematics = KinematicsSolver(model_path)
            
            # Launch viewer and run demo
            print("Launching viewer...")
            with mujoco.viewer.launch_passive(model, data) as viewer_handle:
                self.setup_viewer(model, data, viewer_handle)
                self.execute_planning_loop(model, data, kinematics, viewer_handle)
                
        except FileNotFoundError as e:
            print(f"Error: Could not find required files")
            print(f"   {e}")
            print("   Make sure you're running from the project root directory")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()


class MultiTrajectoryDemo(TrajectoryOptimizationDemo):
    """Base class for demos that generate multiple trajectories (like Pareto search)"""
    
    def __init__(self):
        super().__init__()
        self.trajectories = []
        self.trajectory_colors = []
        
    def get_scene_filename(self) -> str:
        """Override for multi-trajectory scenes"""
        return "multi_trajectory_scene.xml"
        
    def create_scene(self) -> str:
        """Create scene optimized for multiple trajectories"""
        return create_pareto_scene(
            obstacles=self.obstacles,
            target_position=self.target_position,
            max_trajectories=self.get_max_trajectories(),
            output_filename=self.get_scene_filename()
        )
        
    @abstractmethod
    def get_max_trajectories(self) -> int:
        """Get the maximum number of trajectories to generate"""
        pass
        
    def add_trajectory(self, trajectory: List[np.ndarray], color: np.ndarray):
        """Add a trajectory with its visualization color"""
        self.trajectories.append(trajectory)
        self.trajectory_colors.append(color)
        
    def execute_trajectory(self, viewer_handle, model, data, kinematics, trajectory):
        """Override to handle multiple trajectories"""
        from trajectory_visualizer import MultiTrajectoryVisualizer
        
        visualizer = MultiTrajectoryVisualizer(model, data, viewer_handle, kinematics)
        
        # Add all trajectories
        for traj, color in zip(self.trajectories, self.trajectory_colors):
            visualizer.add_trajectory(traj, color)
            
        # Visualize all trajectories
        visualizer.visualize_all_trajectories()
        
        # Run interactive controls
        while True:
            try:
                choice = input("\n[q] - Quit demo: ").strip().lower()
                if choice == 'q':
                    break
                    
                # Keep simulation running
                for _ in range(10):
                    mujoco.mj_step(model, data)
                    viewer_handle.sync()
                    time.sleep(0.01)
                    
            except (KeyboardInterrupt, EOFError):
                break 