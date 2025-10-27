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
import random
import pickle
import json
from datetime import datetime
# Try to import matplotlib for colormap, fallback if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

USE_FAST_PLANNER = True  # Use optimized fast planner (recommended)
SPLINE_BASED = False  # Use B-spline planner (experimental, slower)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.fast_trajopt import FastTrajOptPlanner
from motion_planning.bspline_trajopt import SplineBasedTrajOptPlanner
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
    save_trajectories: bool = True
    experiment_name: str = None


class ParetoSearchDemo(MultiTrajectoryDemo):
    """Pareto search demo with linear weight variation"""
    
    def __init__(self, config: SearchConfiguration):
        super().__init__()
        self.config = config
        self.alpha_values = np.arange(config.alpha_start, config.alpha_end + config.alpha_step, config.alpha_step)
        self.colors = self._generate_plasma_colors(len(self.alpha_values))
        self.results = []
        self.experiment_dir = None
        self.trajectory_metadata = []
        
        # Setup experiment directory if saving is enabled
        if config.save_trajectories:
            self._setup_experiment_directory()
        
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the Pareto search demo"""
        return [
            # Obstacle(center=np.array([0.45, 0.08, 0.2]), radius=0.04, safe_distance=0.04),
            Obstacle(center=np.array([0.45, -0.2, 0.2]), radius=0.04, safe_distance=0.04),
            Obstacle(center=np.array([0.35, 0.06, 0.2]), radius=0.04, safe_distance=0.04),
            # Obstacle(center=np.array([0.35, -0.2, 0.2]), radius=0.04, safe_distance=0.04),
            # Obstacle(center=np.array([-0.65, 0.05, 0.529]), radius=0.04, safe_distance=0.04),
            # Obstacle(center=np.array([-0.65, -0.3, 0.529]), radius=0.04, safe_distance=0.04),
            # Obstacle(center=np.array([-0.55, 0.05, 0.629]), radius=0.05, safe_distance=0.05),
            # Obstacle(center=np.array([-0.55, 0.05, 0.429]), radius=0.05, safe_distance=0.05),
            # PillarObstacle(center=np.array([-0.55, 0.05, 0.629]), radius=0.05, height=1.0, safe_distance=0.05),
            # PillarObstacle(center=np.array([-0.55, 0.05, 0.429]), radius=0.05, height=1.0, safe_distance=0.05)
            # PillarObstacle(center=np.array([-0.6, -0.15, 0.629]), radius=0.04, height=1.0, safe_distance=0.05),
            # PillarObstacle(center=np.array([-0.5, -0.01, 0.629]), radius=0.04, height=1.0, safe_distance=0.05)
        ]
    
    def define_target_position(self) -> np.ndarray:
        """Define target position for the Pareto search demo"""
        # return np.array([-0.7, -0.04, 0.529])
        return np.array([0.65, 0.00, 0.2])
        # return np.array([0.65, 0.03, 0.2])
    
    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the Pareto search demo"""
        # return np.array([0.0, 0.5, 0.0, -2.5, 0.0, 0.45, 1.57])
        # return np.array([0.0, 0.5, 0.0, -2.5, 0.0, -1.0, 1.57])
        return np.array([0.0, -0.5, 0.0, 2.5, 0.0, 1.0, -1.57])
        # return np.array([0.0, -0.35, 3.14, 3.73, 0.0, 5.41, 1.57])
    
    def get_max_trajectories(self) -> int:
        """Get maximum number of trajectories for Pareto search"""
        return len(self.alpha_values)
    
    def get_scene_filename(self) -> str:
        """Get scene filename for Pareto search"""
        return "pareto_search_scene.xml"
        # return "pareto_search_pillar_scene.xml"

    def create_planner(self, model, data):
        """Create constrained trajectory optimization planner"""
        if USE_FAST_PLANNER:
            return FastTrajOptPlanner(
                model, data,
                n_waypoints=25,  # Optimized: fewer waypoints
                dt=0.1,
                max_velocity=1.3,  # Optimized: looser constraints
                max_acceleration=0.7,
                cost_mode='composite',
                cost_sample_rate=2,  # Optimized: sample every 2nd waypoint
                use_global_fk_cache=True  # Optimized: share FK cache across all alpha values
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
            n_waypoints=25,  # Reduced for faster iteration
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
    
    def _setup_experiment_directory(self):
        """Setup experiment directory for saving trajectories and metadata"""
        if self.config.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.experiment_name = f"pareto_search_{timestamp}"
        
        self.experiment_dir = os.path.join("src/pareto_data_and_results", self.config.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Save experiment configuration
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        config_dict = {
            'alpha_start': self.config.alpha_start,
            'alpha_end': self.config.alpha_end,
            'alpha_step': self.config.alpha_step,
            'cost_mode': self.config.cost_mode,
            'rho': self.config.rho,
            'save_trajectories': self.config.save_trajectories,
            'experiment_name': self.config.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'target_position': self.define_target_position().tolist(),
            'start_config': self.define_start_config().tolist(),
            'obstacles': [{'center': obs.center.tolist(), 'radius': obs.radius, 'safe_distance': obs.safe_distance} 
                         for obs in self.obstacles]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Experiment directory created: {self.experiment_dir}")
    
    def _save_trajectory(self, trajectory: List[np.ndarray], alpha: float, 
                        length_cost: float, obstacle_cost: float, color: np.ndarray,
                        optimization_metadata: dict = None):
        """Save a single trajectory with its metadata"""
        if not self.config.save_trajectories or self.experiment_dir is None:
            return
        
        trajectory_id = f"alpha_{alpha:.3f}".replace('.', 'p')
        trajectory_path = os.path.join(self.experiment_dir, f"trajectory_{trajectory_id}.pkl")
        
        # Convert trajectory to numpy array for easier storage
        trajectory_array = np.array(trajectory)
        
        trajectory_data = {
            'trajectory': trajectory_array,
            'alpha': alpha,
            'length_cost': length_cost,
            'obstacle_cost': obstacle_cost,
            'color': color,
            'length_weight': alpha,
            'obstacle_weight': 1.0 - alpha,
            'trajectory_id': trajectory_id,
            'waypoint_count': len(trajectory),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optimization metadata if provided
        if optimization_metadata:
            trajectory_data['optimization'] = optimization_metadata
        
        # Save trajectory data
        with open(trajectory_path, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        # Add to metadata list
        metadata_entry = {
            'trajectory_id': trajectory_id,
            'alpha': alpha,
            'length_cost': length_cost,
            'obstacle_cost': obstacle_cost,
            'length_weight': alpha,
            'obstacle_weight': 1.0 - alpha,
            'filename': f"trajectory_{trajectory_id}.pkl",
            'waypoint_count': len(trajectory),
            'color': color.tolist()
        }
        
        # Add optimization info to metadata
        if optimization_metadata:
            metadata_entry['iterations'] = optimization_metadata.get('iterations', 0)
            metadata_entry['final_optimization_cost'] = optimization_metadata.get('final_optimization_cost', 0.0)
            metadata_entry['cost_mode'] = optimization_metadata.get('cost_mode', self.config.cost_mode)
        
        self.trajectory_metadata.append(metadata_entry)
        
        print(f"Saved trajectory for α={alpha:.3f} to {trajectory_path}")
    
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
            # planner.enable_fixed_z_constraint(
            #     kinematics_solver=kinematics,
            #     target_z=self.define_target_position()[2],
            #     tol=0.05  # meters; tighten/loosen as needed
            # )
            
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

            # z_constraint = FixedZCostFunction(
            #     kinematics_solver=kinematics,
            #     target_z=self.define_target_position()[2],  # or hardcode like 0.529
            #     weight=100.0  # Large enough to enforce it as a constraint
            # )

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
            trajectory, success, optimization_metadata = planner.plan(self.start_config, goal_config)
            
            if success:
                self.add_trajectory(trajectory, color)

                trajectory_np = np.array(trajectory)  # shape: (N_waypoints, DOF)

                f_length = length_cost.compute_cost(trajectory_np)
                f_obstacle = safety_cost.compute_cost(trajectory_np)
                print(f"Length cost for α={alpha:.1f}: {f_length:.4f}")
                print(f"Closeness cost for α={alpha:.1f}: {f_obstacle:.4f}")

                self.results.append((f_length, f_obstacle, alpha))
                
                # Save trajectory if enabled (with optimization metadata)
                self._save_trajectory(trajectory, alpha, f_length, f_obstacle, color, optimization_metadata)
                
                # Optionally print timing summary for first trajectory
                if len(self.results) == 1 and 'timing' in optimization_metadata:
                    print(f"\nTiming breakdown for first trajectory (α={alpha:.1f}):")
                    planner.print_timing_summary(optimization_metadata['timing'])
    
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
        from motion_planning.fast_trajopt import CachedKinematicsSolver
        
        print(f"Starting Linear Weight Search")
        print(f"Cost formulation: {self.config.cost_mode.upper()}")
        print(f"Alpha range: [{self.config.alpha_start:.1f}, {self.config.alpha_end:.1f}] step {self.config.alpha_step:.1f}")
        
        # Clear global FK cache at the start of a new search
        if USE_FAST_PLANNER:
            CachedKinematicsSolver.clear_global_cache()
            print("  Global FK cache cleared for fresh Pareto search")
        
        successful_count = 0
        
        for i, alpha in enumerate(self.alpha_values):
            color = self.colors[i]
            trajectory = self.plan_single_trajectory(float(alpha), color, model, data, kinematics)
            if trajectory is not None:
                successful_count += 1
        
        print(f"\nSearch complete: {successful_count}/{len(self.alpha_values)} successful trajectories")
        
        # Print final global FK cache statistics
        if USE_FAST_PLANNER:
            global_stats = CachedKinematicsSolver.get_global_cache_stats()
            print(f"\n=== Global FK Cache Summary (All Optimizations) ===")
            print(f"  Total hits: {global_stats['hits']:,}")
            print(f"  Total misses: {global_stats['misses']:,}")
            print(f"  Overall hit rate: {global_stats['hit_rate']*100:.1f}%")
            print(f"  Final cache size: {global_stats['cache_size']} entries")
            total = global_stats['hits'] + global_stats['misses']
            if total > 0:
                saved = global_stats['hits']
                print(f"  Total FK calls saved: {saved:,} ({saved/total*100:.1f}% reduction)")
        
        # Save trajectory metadata if trajectories were saved
        if self.config.save_trajectories and self.experiment_dir is not None:
            self._save_trajectory_metadata()

    def _save_trajectory_metadata(self):
        """Save metadata for all trajectories to a JSON file"""
        if not self.trajectory_metadata:
            return
            
        metadata_path = os.path.join(self.experiment_dir, "trajectory_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.trajectory_metadata, f, indent=2)
        
        print(f"Trajectory metadata saved to {metadata_path}")

    def save_results_to_csv(self, output_dir="src/pareto_data_and_results", filename="tradeoff_data.csv"):
        if not self.results:
            print("No results to save.")
            return
        os.makedirs(output_dir, exist_ok=True)
        
        full_path = os.path.join(output_dir, filename)

        with open(full_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["length", "closeness", "alpha"])
            writer.writerows(self.results)

        print(f"Results saved to {full_path}")
    
    def execute_planning_loop(self, model, data, kinematics, viewer_handle):
        """Execute Pareto search and visualize results"""
        # Run the search
        self.run_pareto_search(model, data, kinematics)
        
        # Plot trade-off between length and obstacle cost
        self.save_results_to_csv()  # Use default output_dir and filename

        # Visualize all trajectories
        if self.trajectories:
            self.execute_trajectory(viewer_handle, model, data, kinematics, None)
        else:
            print("No successful trajectories to visualize")


def parse_arguments():
    """Parse command line arguments for cost mode and search parameters"""
    parser = argparse.ArgumentParser(description='Linear Weight Search for Trajectory Optimization Pareto Analysis')
    
    parser.add_argument('--cost-mode', choices=['sum', 'max', 'max_constrained'], default='sum',
                       help='Cost function formulation (default: sum)')
    parser.add_argument('--rho', type=float, default=0.01,
                       help='Tie-breaking parameter for max mode (default: 0.01)')
    parser.add_argument('--alpha-start', type=float, default=0.0,
                       help='Start value for alpha parameter (default: 0.0)')
    parser.add_argument('--alpha-end', type=float, default=1.0,
                       help='End value for alpha parameter (default: 1.0)')
    parser.add_argument('--alpha-step', type=float, default=0.1,
                       help='Step size for alpha parameter (default: 0.1)')
    # arguments related to data saving 
    parser.add_argument('--csv-file', type=str, default='tradeoff_data.csv',
                    help='Name of the CSV file to save results (default: tradeoff_data_100_sphere_sum.csv)')
    parser.add_argument('--output-dir', type=str, default='src/pareto_data_and_results',
                    help='Directory to save CSV file (default: src/pareto_data_and_results)')

    # NEW: seed argument
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    
    # Trajectory saving arguments
    parser.add_argument('--save-trajectories', action='store_true', default=True,
                        help='Save optimized trajectories to experiment directory (default: True)')
    parser.add_argument('--no-save-trajectories', dest='save_trajectories', action='store_false',
                        help='Disable trajectory saving')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for the experiment directory (default: auto-generated with timestamp)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # --- NEW: set seeds ---
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed)

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
        rho=args.rho,
        save_trajectories=args.save_trajectories,
        experiment_name=args.experiment_name
    )
    
    # Create and run demo
    demo = ParetoSearchDemo(config)
    demo.run_demo()
