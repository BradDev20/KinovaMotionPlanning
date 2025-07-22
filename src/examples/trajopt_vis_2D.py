#!/usr/bin/env python3
"""
2D TrajOpt Visualization Demo

This demo demonstrates trajectory optimization with co-planar obstacles/goals
and multiple trajectory visualization with accumulated traces.
"""

import sys
import os
import numpy as np
import random
from typing import List

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.unconstrained_trajopt import UnconstrainedTrajOptPlanner
from motion_planning.utils import Obstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    VelocityCostFunction,
    AccelerationCostFunction,
    FixedZCostFunction
)
from trajectory_optimization_demo import TrajectoryOptimizationDemo
from trajectory_visualizer import TrajectoryVisualizationManager


class TrajOpt2DVisualizationDemo(TrajectoryOptimizationDemo):
    """2D trajectory optimization demo with multi-run visualization"""
    
    def __init__(self):
        super().__init__()
        self.run_counter = 0
    
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the 2D visualization demo"""
        return [
            Obstacle(center=np.array([-0.6, -0.05, 0.529]), radius=0.08, safe_distance=0.05),
            Obstacle(center=np.array([-0.6, 0.15, 0.529]), radius=0.08, safe_distance=0.05),
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position for the 2D visualization demo"""
        return np.array([-0.8, 0.1, 0.529])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the 2D visualization demo"""
        return np.array([0.0, 0.5, 0.0, -2.5, 0.0, 0.45, 1.57])
    
    def get_scene_filename(self) -> str:
        """Get scene filename for 2D visualization demo"""
        return "trajopt_multi_obstacle_scene.xml"
    
    def get_trace_dot_count(self) -> int:
        """Get larger number of trace dots for multiple trajectories"""
        return 500
    
    def create_planner(self, model, data):
        """Create unconstrained trajectory optimization planner"""
        return UnconstrainedTrajOptPlanner(model, data, n_waypoints=50, dt=0.1)
    
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for 2D visualization"""
        try:
            if hasattr(self, '_strategy_choice') and self._strategy_choice == '1':
                # RISKY Strategy: Prioritize short paths
                length_cost = TrajectoryLengthCostFunction(weight=10.0)
                planner.add_cost_function(length_cost)
                safety_weight = 0.1
            else:
                # SAFE Strategy: Prioritize safety (default)
                length_cost = TrajectoryLengthCostFunction(weight=0.01)
                planner.add_cost_function(length_cost)
                
                safety_cost = ObstacleAvoidanceCostFunction(
                    kinematics_solver=kinematics,
                    obstacles=self.obstacles,
                    weight=300.0,
                )
                planner.add_cost_function(safety_cost)
                safety_weight = 300.0
                
        except (KeyboardInterrupt, EOFError):
            # Default to safe strategy
            length_cost = TrajectoryLengthCostFunction(weight=1.0)
            planner.add_cost_function(length_cost)
            
            safety_cost = ObstacleAvoidanceCostFunction(
                kinematics_solver=kinematics,
                obstacles=self.obstacles,
                weight=20.0,
            )
            planner.add_cost_function(safety_cost)
            safety_weight = 20.0
        
        # Multi-obstacle avoidance
        obstacle_cost = ObstacleAvoidanceCostFunction(
            kinematics_solver=kinematics,
            obstacles=self.obstacles,
            weight=200.0
        )
        planner.add_cost_function(obstacle_cost)
        
        # Velocity and acceleration smoothness
        velocity_cost = VelocityCostFunction(weight=2)
        planner.add_cost_function(velocity_cost)
        
        acceleration_cost = AccelerationCostFunction(weight=2)
        planner.add_cost_function(acceleration_cost)
        
        # Fixed Z cost function
        fixed_z_cost = FixedZCostFunction(
            kinematics_solver=kinematics,
            target_z=0.529,
            weight=10000.0
        )
        planner.add_cost_function(fixed_z_cost)
        
        return True
    
    def execute_trajectory(self, viewer_handle, model, data, kinematics, trajectory):
        """Execute trajectory with custom color for multi-run visualization"""
        # Generate random color for this run
        trace_color = np.array([random.random(), random.random(), random.random(), 0.8])
        
        # Create trajectory visualizer
        visualizer = TrajectoryVisualizationManager(model, data, viewer_handle, kinematics)
        
        # Execute trajectory with custom color
        print(f"Executing trajectory run {self.run_counter + 1}...")
        visualizer.execute_trajectory(
            trajectory=trajectory,
            trace_color=trace_color,
            verbose=False
        )
        
        self.run_counter += 1
    
    def execute_planning_loop(self, model, data, kinematics, viewer_handle):
        """Custom planning loop with multiple trajectory support"""
        while True:
            # Strategy choice before each plan
            print("\nChoose planning strategy:")
            print("  [1] RISKY: Trajectory length minimization (threads between obstacles)")
            print("  [2] SAFE: Safety importance (goes around obstacles)")
            
            self._strategy_choice = input("Enter choice (1 or 2, default=2): ").strip()
            
            # Plan trajectory
            trajectory = self.plan_trajectory(model, data, kinematics)
            
            if trajectory is not None:
                self.execute_trajectory(viewer_handle, model, data, kinematics, trajectory)
            else:
                print("Cannot proceed with visualization due to planning failure.")
            
            # Ask to continue or exit
            while True:
                choice = input("\nDo you want to plan another trajectory? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    break  # Run another iteration
                elif choice in ['n', 'no']:
                    print("Exiting the demo...")
                    return  # Exit
                else:
                    print("Please enter 'y' or 'n'.")


if __name__ == "__main__":
    demo = TrajOpt2DVisualizationDemo()
    demo.run_demo() 