#!/usr/bin/env python3
"""
Single Trajectory Optimization Demo

This script optimizes a single trajectory for a given alpha value (weight between 
trajectory length and obstacle avoidance).
"""

import sys
import os
import numpy as np
import argparse
from typing import List

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from motion_planning.bspline_trajopt import SplineBasedTrajOptPlanner
from motion_planning.fast_trajopt import FastTrajOptPlanner
from motion_planning.utils import Obstacle, PillarObstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    FixedZCostFunction,
    CompositeCostFunction,
    CostModeFactory
)
from trajectory_optimization_demo import TrajectoryOptimizationDemo

Z_CONSTRAINT = False
USE_FAST_PLANNER = False  # Use optimized fast planner (recommended)
SPLINE_BASED = False  # Use B-spline planner (experimental, slower)

class SingleTrajectoryDemo(TrajectoryOptimizationDemo):
    """Demo that optimizes a single trajectory for a given alpha value"""
    
    def __init__(self, alpha: float = 0.5, cost_mode: str = 'sum', rho: float = 0.01):
        """
        Initialize the single trajectory demo
        
        Args:
            alpha: Weight for trajectory length (0-1). obstacle weight = 1 - alpha
            cost_mode: Cost function formulation ('sum', 'max', or 'max_constrained')
            rho: Tie-breaking parameter for max mode
        """
        self.alpha = alpha
        self.cost_mode = cost_mode
        self.rho = rho
        super().__init__()
        
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the demo"""
        return [
            # Obstacle(center=np.array([0.45, -0.2, 0.2]), radius=0.04, safe_distance=0.04),
            Obstacle(center=np.array([0.35, 0.06, 0.2]), radius=0.04, safe_distance=0.04),
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position for the demo"""
        return np.array([0.65, 0.00, 0.2])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the demo"""
        return np.array([0.0, -0.5, 0.0, 2.5, 0.0, 1.0, -1.57])
    
    def get_scene_filename(self) -> str:
        """Get scene filename"""
        return "pareto_search_scene.xml"

    def create_planner(self, model, data):
        """Create trajectory optimization planner"""
        if USE_FAST_PLANNER:
            return FastTrajOptPlanner(
                model, data,
                n_waypoints=25,  # Optimized: fewer waypoints
                dt=0.1,
                max_velocity=1.3,  # Optimized: looser constraints
                max_acceleration=0.7,
                cost_mode='composite',
                cost_sample_rate=2  # Optimized: sample every 2nd waypoint
            )
        elif SPLINE_BASED:
            return SplineBasedTrajOptPlanner(
                model, data, 
                n_waypoints=12,
                dt=0.1,
                max_velocity=1.0,
                max_acceleration=0.7,
                cost_mode='composite'
            )
        else:
            return ConstrainedTrajOptPlanner(
                model, data, 
                n_waypoints=25,
                dt=0.1,
                max_velocity=1.0,       
                max_acceleration=0.7,
                cost_mode='composite'
            )
    
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions with the specified alpha value"""
        try:
            # Calculate weights
            length_weight = self.alpha
            obstacle_weight = 1.0 - self.alpha
            
            print(f"\nSetting up cost functions:")
            print(f"  Alpha (length weight): {length_weight:.2f}")
            print(f"  Obstacle weight: {obstacle_weight:.2f}")
            print(f"  Cost mode: {self.cost_mode.upper()}")
            if self.cost_mode == 'max':
                print(f"  Rho (tie-breaking): {self.rho}")

            if Z_CONSTRAINT:
                planner.enable_fixed_z_constraint(
                    kinematics_solver=kinematics,
                    target_z=self.define_target_position()[2],
                    tol=0.05  # meters; tighten/loosen as needed
                )

            # Create individual cost functions
            length_cost = TrajectoryLengthCostFunction(
                kinematics_solver=kinematics,
                weight=1.0,
                normalization_bounds=(0.0, 1.0)
            )

            safety_cost = ObstacleAvoidanceCostFunction(
                kinematics_solver=kinematics,
                obstacles=self.obstacles,
                weight=1.0,
                normalization_bounds=(0.0, 1.0),
                decay_rate=15.0,
                bias=-0.08,
                aggregate="avg"
            )

            # Set up composite cost function
            cost_functions = [length_cost, safety_cost]
            weights = [length_weight, obstacle_weight]
            
            composite_cost = planner.setup_composite_cost(
                cost_functions=cost_functions,
                weights=weights,
                formulation=self.cost_mode,
                rho=self.rho
            )
            
            # Store cost functions for later evaluation
            self.length_cost = length_cost
            self.safety_cost = safety_cost
            
            return True
            
        except Exception as e:
            print(f"Error setting up cost functions: {e}")
            return False
    
    def plan_trajectory(self, model, data, kinematics):
        """Plan a single trajectory"""
        print(f"\nPlanning trajectory for α={self.alpha:.2f}")
        
        # Setup IK
        goal_config = self.solve_inverse_kinematics(kinematics)
        if goal_config is None:
            return None
        
        # Create planner
        planner = self.create_planner(model, data)
        
        # Setup cost functions
        if not self.setup_cost_functions(planner, kinematics):
            return None
        
        # Plan trajectory
        print("\nOptimizing trajectory...")
        trajectory, success, optimization_metadata = planner.plan(self.start_config, goal_config)
        
        if success:
            trajectory_np = np.array(trajectory)
            
            # Compute and display costs
            f_length = self.length_cost.compute_cost(trajectory_np)
            f_obstacle = self.safety_cost.compute_cost(trajectory_np)
            
            print(f"\n✓ Optimization successful!")
            print(f"  Length cost: {f_length:.4f}")
            print(f"  Obstacle cost: {f_obstacle:.4f}")
            print(f"  Iterations: {optimization_metadata.get('iterations', 'N/A')}")
            print(f"  Final optimization cost: {optimization_metadata.get('final_optimization_cost', 'N/A'):.4f}")
            
            # Print timing summary if available
            if 'timing' in optimization_metadata:
                planner.print_timing_summary(optimization_metadata['timing'])
            
            return trajectory
        else:
            print(f"\n✗ Optimization failed")
            return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Single Trajectory Optimization Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Equal weighting between length and obstacle avoidance
  python single_traj.py --alpha 0.5
  
  # Prioritize short trajectories
  python single_traj.py --alpha 0.8
  
  # Prioritize obstacle avoidance
  python single_traj.py --alpha 0.2
  
  # Use max formulation instead of sum
  python single_traj.py --alpha 0.5 --cost-mode max
        """
    )
    
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for trajectory length (0-1). Obstacle weight = 1 - alpha (default: 0.5)')
    parser.add_argument('--cost-mode', choices=['sum', 'max', 'max_constrained'], default='sum',
                       help='Cost function formulation (default: sum)')
    parser.add_argument('--rho', type=float, default=0.01,
                       help='Tie-breaking parameter for max mode (default: 0.01)')
    
    args = parser.parse_args()
    
    # Validate alpha
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("Alpha must be between 0.0 and 1.0")
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    print("="*60)
    print("Single Trajectory Optimization Demo")
    print("="*60)
    print(f"Alpha (length weight): {args.alpha:.2f}")
    print(f"Obstacle weight: {1.0 - args.alpha:.2f}")
    print(f"Cost mode: {args.cost_mode.upper()}")
    if args.cost_mode == 'max':
        print(f"Rho: {args.rho}")
    print("="*60)
    
    # Create and run demo
    demo = SingleTrajectoryDemo(
        alpha=args.alpha,
        cost_mode=args.cost_mode,
        rho=args.rho
    )
    demo.run_demo()
