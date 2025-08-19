#!/usr/bin/env python3
"""
Constrained TrajOpt Demo

This demo demonstrates constrained trajectory optimization with multi-obstacle 
avoidance using velocity and acceleration constraints.
"""

import sys
import os
import numpy as np
from typing import List

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from motion_planning.utils import Obstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
)
from trajectory_optimization_demo import TrajectoryOptimizationDemo


class ConstrainedOptimDemo(TrajectoryOptimizationDemo):
    """Constrained trajectory optimization demo with multi-obstacle avoidance"""
    
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the constrained optimization demo"""
        return [
            Obstacle(center=np.array([0.55, -0.1, 0.75]), radius=0.08),
            Obstacle(center=np.array([0.55, -0.45, 0.65]), radius=0.08),
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position for the constrained optimization demo"""
        return np.array([0.6, -0.3, 0.4])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the constrained optimization demo"""
        return np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    
    def get_scene_filename(self) -> str:
        """Get scene filename for constrained optimization demo"""
        return "trajopt_constrained_scene.xml"
    
    def create_planner(self, model, data):
        """Create constrained trajectory optimization planner"""
        return ConstrainedTrajOptPlanner(
            model, 
            data, 
            n_waypoints=50, 
            dt=0.1,
            max_velocity=1.0,
            max_acceleration=0.7
        )
    
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for constrained optimization"""
        try:
            strategy_choice = input("Enter choice (risky=1 or safe=2): ").strip()
            
            if strategy_choice == '1':
                print("RISKY Strategy Selected: Prioritizing trajectory length")
                length_weight = 1.0
                safety_weight = 0.0
            else:
                print("SAFE Strategy Selected: Prioritizing safety")
                length_weight = 0.0
                safety_weight = 1.0

            # Add trajectory length cost function
            length_cost = TrajectoryLengthCostFunction(
                kinematics_solver=kinematics,
                weight=length_weight,
                normalization_bounds=(1.0, 2.0)
            )
            planner.add_cost_function(length_cost)

            # Add obstacle avoidance cost function
            safety_cost = ObstacleAvoidanceCostFunction(
                kinematics_solver=kinematics,
                obstacles=self.obstacles,
                weight=safety_weight,
                normalization_bounds=(10.0, 20.0)
            )
            planner.add_cost_function(safety_cost)
            
            return True
            
        except Exception as e:
            print(f"Error setting up cost functions: {e}")
            return False


if __name__ == "__main__":
    demo = ConstrainedOptimDemo()
    demo.run_demo() 