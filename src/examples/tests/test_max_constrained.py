#!/usr/bin/env python3
"""
Max Constrained Mode Test - Epigraph Reformulation Demo

This demo tests the epigraph reformulation (max_constrained mode) that moves
the max operator from the cost function to constraints:

Original:    min_T[max_i(w_i * f_i(T)) + ρ * Σf_i(T)]
Epigraph:    min_{T,t}[t + ρ * Σf_i(T)] s.t. w_i * f_i(T) ≤ t
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
    CompositeCostFunction,
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    VelocityCostFunction
)
from trajectory_optimization_demo import TrajectoryOptimizationDemo


class MaxConstrainedDemo(TrajectoryOptimizationDemo):
    """Demo comparing 'max' mode vs 'max_constrained' epigraph reformulation"""
    
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the demo"""
        return [
            Obstacle(center=np.array([0.55, -0.1, 0.75]), radius=0.08),
            Obstacle(center=np.array([0.55, -0.45, 0.65]), radius=0.08),
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position"""
        return np.array([0.6, -0.3, 0.4])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration"""
        return np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    
    def get_scene_filename(self) -> str:
        """Get scene filename"""
        return "test_max_constrained_scene.xml"
    
    def create_planner(self, model, data):
        """Create constrained trajectory optimization planner"""
        return ConstrainedTrajOptPlanner(
            model, data,
            n_waypoints=20,
            dt=0.1,
            max_velocity=2.0,
            max_acceleration=10.0,
            cost_mode='composite'
        )
    
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for the planner"""
        print("\n" + "=" * 80)
        print("Choose optimization mode:")
        print("  [1] Standard 'max' mode (baseline)")
        print("  [2] Epigraph 'max_constrained' mode (reformulation)")
        print("=" * 80)
        
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            
            # Create individual cost functions
            length_cost = TrajectoryLengthCostFunction(kinematics, weight=1.0)
            obstacle_cost = ObstacleAvoidanceCostFunction(
                kinematics, 
                self.obstacles, 
                weight=1.0,
                aggregate='sum',
                decay_rate=5.0
            )
            velocity_cost = VelocityCostFunction(weight=1.0, max_velocity=2.0)
            
            if choice == '1':
                # Standard max mode
                print("\n📊 Using standard 'max' mode")
                composite = CompositeCostFunction(
                    cost_functions=[length_cost, obstacle_cost, velocity_cost],
                    weights=[1.0, 2.0, 0.5],
                    mode='max',
                    rho=0.01
                )
            elif choice == '2':
                # Epigraph reformulation mode
                print("\nUsing epigraph 'max_constrained' mode")
                print("   Formulation: min_{T,t}[t + ρ*Σf_i(T)] s.t. w_i*f_i(T) ≤ t")
                composite = CompositeCostFunction(
                    cost_functions=[length_cost, obstacle_cost, velocity_cost],
                    weights=[1.0, 2.0, 0.5],
                    mode='max_constrained',
                    rho=0.01
                )
            else:
                print("Invalid choice")
                return False
            
            planner.composite_cost_function = composite
            return True
            
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nSetup cancelled")
            return False


if __name__ == "__main__":
    demo = MaxConstrainedDemo()
    demo.run_demo() 