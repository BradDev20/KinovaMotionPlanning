#!/usr/bin/env python3
"""
Unconstrained TrajOpt Demo

This demo demonstrates unconstrained trajectory optimization with multi-obstacle 
avoidance using cost function-based smoothness control (velocity and acceleration).
"""

import sys
import os
import numpy as np
from typing import List

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.unconstrained_trajopt import UnconstrainedTrajOptPlanner
from motion_planning.utils import Obstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    VelocityCostFunction,
    AccelerationCostFunction
)
from trajectory_optimization_demo import TrajectoryOptimizationDemo


class UnconstrainedOptimDemo(TrajectoryOptimizationDemo):
    """Unconstrained trajectory optimization demo with multi-obstacle avoidance"""
    
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the unconstrained optimization demo"""
        return [
            Obstacle(center=np.array([0.55, -0.10, 0.85]), radius=0.12),
            Obstacle(center=np.array([0.55, -0.35, 0.65]), radius=0.12),
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position for the unconstrained optimization demo"""
        return np.array([0.6, -0.3, 0.4])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the unconstrained optimization demo"""
        return np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    
    def get_scene_filename(self) -> str:
        """Get scene filename for unconstrained optimization demo"""
        return "trajopt_unconstrained_scene.xml"
    
    def create_planner(self, model, data):
        """Create unconstrained trajectory optimization planner"""
        return UnconstrainedTrajOptPlanner(model, data, n_waypoints=50, dt=0.1)
    
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for unconstrained optimization"""
        print("Choose planning strategy:")
        print("  [1] RISKY: Trajectory length minimization (threads between obstacles)")
        print("  [2] SAFE: Safety importance (goes around obstacles)")
        
        try:
            strategy_choice = input("Enter choice (1 or 2, default=2): ").strip()
            
            if strategy_choice == '1':
                print("RISKY Strategy Selected: Prioritizing trajectory length")
                length_cost = TrajectoryLengthCostFunction(weight=10.0)
                planner.add_cost_function(length_cost)
            else:
                print("SAFE Strategy Selected: Prioritizing safety importance")
                # Trajectory length with low weight
                length_cost = TrajectoryLengthCostFunction(weight=0.001)
                planner.add_cost_function(length_cost)
                
                # High-weight obstacle avoidance
                safety_cost = ObstacleAvoidanceCostFunction(
                    kinematics_solver=kinematics,
                    obstacles=self.obstacles,
                    weight=400.0,
                    decay_rate=10,
                )
                planner.add_cost_function(safety_cost)
            
            # Velocity smoothness
            velocity_cost = VelocityCostFunction(weight=3)
            planner.add_cost_function(velocity_cost)
            
            # Acceleration smoothness
            acceleration_cost = AccelerationCostFunction(weight=3)
            planner.add_cost_function(acceleration_cost)
            
            return True
            
        except Exception as e:
            print(f"Error setting up cost functions: {e}")
            return False


if __name__ == "__main__":
    demo = UnconstrainedOptimDemo()
    demo.run_demo() 