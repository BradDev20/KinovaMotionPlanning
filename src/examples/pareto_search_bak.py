#!/usr/bin/env python3
"""
Linear Weight Search for Trajectory Optimization Pareto Analysis

This script performs a linear search over weights for trajectory length and obstacle avoidance,
visualizing all resulting trajectories simultaneously with a plasma colormap.
"""

import sys
import os
import numpy as np
import argparse
from typing import List
from dataclasses import dataclass
import csv

# Try to import matplotlib for colormap, fallback if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from motion_planning.utils import Obstacle, PillarObstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    FixedZCostFunction,
    CompositeCostFunction,
    CostModeFactory
)
from trajectory_optimization_demo import MultiTrajectoryDemo


@dataclass
class SearchConfiguration:
    """Configuration for the linear weight search"""
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_step: float = 1.0
    cost_mode: str = 'sum'
    rho: float = 0.01


class ParetoSearchDemo(MultiTrajectoryDemo):
    """Pareto search demo with linear weight variation"""
    
    def __init__(self, config: SearchConfiguration):
        super().__init__()
        self.config = config
        self.alpha_values = np.arange(config.alpha_start, config.alpha_end + config.alpha_step, config.alpha_step)
        self.colors = self._generate_plasma_colors(len(self.alpha_values))
        self.results = []
        
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the Pareto search demo"""
        return [
            # Obstacle(center=np.array([-0.55, 0.05, 0.629]), radius=0.05, safe_distance=0.05),
            # Obstacle(center=np.array([-0.55, 0.05, 0.429]), radius=0.05, safe_distance=0.05),
            PillarObstacle(center=np.array([-0.55, 0.05, 0.629]), radius=0.05, height=1.0, safe_distance=0.05),
            PillarObstacle(center=np.array([-0.55, 0.05, 0.429]), radius=0.05, height=1.0, safe_distance=0.05)
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position for the Pareto search demo"""
        return np.array([-0.7, 0.1, 0.529])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the Pareto search demo"""
        return np.array([0.0, 0.5, 0.0, -2.5, 0.0, 0.45, 1.57])
    
    def get_max_trajectories(self) -> int:
        """Get maximum number of trajectories for Pareto search"""
        return len(self.alpha_values)
    
    def get_scene_filename(self) -> str:
        """Get scene filename for Pareto search"""
        # return "pareto_search_scene.xml"
        return "pareto_search_pillar_scene.xml"

    def create_planner(self, model, data):
        """Create constrained trajectory optimization planner"""
        return ConstrainedTrajOptPlanner(
            model, data, 
            n_waypoints=50,
            dt=0.1,
            max_velocity=1.0,
            max_acceleration=0.7,
            cost_mode='composite'
        )
    
    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for current alpha value"""
        # This will be called for each alpha value
        return True  # Actual setup happens in plan_single_trajectory
    
    def _generate_plasma_colors(self, n_colors: int) -> np.ndarray:
        """Generate plasma colormap colors for the trajectories"""
        if HAS_MATPLOTLIB:
            plasma = plt.cm.plasma
            color_indices = np.linspace(0, 1, n_colors)
            colors = []
            for idx in color_indices:
                rgba = plasma(idx)
                colors.append([rgba[0], rgba[1], rgba[2], 0.8])
            return np.array(colors)
        else:
            # Fallback: simple interpolation from purple to yellow
            colors = []
            for i in range(n_colors):
                t = i / (n_colors - 1) if n_colors > 1 else 0
                r = 0.5 + 0.5 * t
                g = 0.0 + 1.0 * t
                b = 1.0 - 1.0 * t
                colors.append([r, g, b, 0.8])
            return np.array(colors)
    
    def plan_single_trajectory(self, alpha: float, color: np.ndarray, model, data, kinematics):
        """Plan a single trajectory for given alpha value"""
        print(f"Planning trajectory for α={alpha:.1f} ({self.config.cost_mode.upper()} mode)")
        
        # Calculate weights
        length_weight = alpha
        obstacle_weight = 1.0 - alpha
        
        # Setup IK
        goal_config = self.solve_inverse_kinematics(kinematics)
        if goal_config is None:
            return None
        
        # Create planner with composite cost mode
        planner = self.create_planner(model, data)
        
        try:
            # Create individual cost functions
            length_cost = TrajectoryLengthCostFunction(
                kinematics_solver=kinematics,
                weight=1.0,
                normalization_bounds=(1.0, 2.0)
            )

            safety_cost = ObstacleAvoidanceCostFunction(
                kinematics_solver=kinematics,
                obstacles=self.obstacles,
                weight=1.0,
                normalization_bounds=(0.0, 1.0),
                decay_rate=5.0
            )

            # Set up composite cost function
            cost_functions = [length_cost, safety_cost]
            weights = [length_weight, obstacle_weight]
            
            composite_cost = planner.setup_composite_cost(
                cost_functions=cost_functions,
                weights=weights,
                formulation=self.config.cost_mode,
                rho=self.config.rho
            )

            # Plan trajectory
            trajectory, success = planner.plan(self.start_config, goal_config)
            
            if success:
                self.add_trajectory(trajectory, color)

                trajectory_np = np.array(trajectory)  # shape: (N_waypoints, DOF)

                f_length = length_cost.compute_cost(trajectory_np)
                f_obstacle = safety_cost.compute_cost(trajectory_np)
                print(f"Length cost for α={alpha:.1f}: {f_length:.4f}")
                print(f"Closeness cost for α={alpha:.1f}: {f_obstacle:.4f}")

                self.results.append((f_length, f_obstacle, alpha))
    
                print(f"α={alpha:.1f}: Success")
                return trajectory
            else:
                print(f"α={alpha:.1f}: Failed")
                return None
                
        except Exception as e:
            print(f"α={alpha:.1f}: Error - {e}")
            return None
    
    def run_pareto_search(self, model, data, kinematics):
        """Run the complete Pareto search"""
        print(f"Starting Linear Weight Search")
        print(f"Cost formulation: {self.config.cost_mode.upper()}")
        print(f"Alpha range: [{self.config.alpha_start:.1f}, {self.config.alpha_end:.1f}] step {self.config.alpha_step:.1f}")
        
        successful_count = 0
        
        for i, alpha in enumerate(self.alpha_values):
            color = self.colors[i]
            trajectory = self.plan_single_trajectory(float(alpha), color, model, data, kinematics)
            if trajectory is not None:
                successful_count += 1
        
        print(f"Search complete: {successful_count}/{len(self.alpha_values)} successful trajectories")

    def save_results_to_csv(self, filename="tradeoff_data_100_w_sum.csv"):
        if not self.results:
            print("No results to save.")
            return
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["length", "closeness", "alpha"])
            writer.writerows(self.results)
        print(f"Results saved to {filename}")
    
    def execute_planning_loop(self, model, data, kinematics, viewer_handle):
        """Execute Pareto search and visualize results"""
        # Run the search
        self.run_pareto_search(model, data, kinematics)
        
        # Plot trade-off between length and obstacle cost
        self.save_results_to_csv()

        # Visualize all trajectories
        if self.trajectories:
            self.execute_trajectory(viewer_handle, model, data, kinematics, None)
        else:
            print("No successful trajectories to visualize")


def parse_arguments():
    """Parse command line arguments for cost mode and search parameters"""
    parser = argparse.ArgumentParser(description='Linear Weight Search for Trajectory Optimization Pareto Analysis')
    
    parser.add_argument('--cost-mode', choices=['sum', 'max'], default='sum',
                       help='Cost function formulation (default: sum)')
    parser.add_argument('--rho', type=float, default=1e-7,
                       help='Tie-breaking parameter for max mode (default: 0.01)')
    parser.add_argument('--alpha-start', type=float, default=0.0,
                       help='Start value for alpha parameter (default: 0.0)')
    parser.add_argument('--alpha-end', type=float, default=1.0,
                       help='End value for alpha parameter (default: 1.0)')
    parser.add_argument('--alpha-step', type=float, default=0.1,
                       help='Step size for alpha parameter (default: 0.1)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("Linear Weight Search for Trajectory Optimization")
    print(f"Cost Mode: {args.cost_mode.upper()}")
    if args.cost_mode == 'max':
        print(f"Tie-breaking ρ: {args.rho}")
    print(f"Alpha range: [{args.alpha_start:.1f}, {args.alpha_end:.1f}] step {args.alpha_step:.1f}")
    
    # Create configuration
    config = SearchConfiguration(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_step=args.alpha_step,
        cost_mode=args.cost_mode,
        rho=args.rho
    )
    
    # Create and run demo
    demo = ParetoSearchDemo(config)
    demo.run_demo()
