#!/usr/bin/env python3
"""
Linear Weight Search for Trajectory Optimization Pareto Analysis

This script performs a linear search over weights for trajectory length and safety,
visualizing all resulting trajectories simultaneously with a plasma colormap.

Tasks:
- balls  : length vs obstacle avoidance (your original)
- shelf  : length vs side-wall risk inside a backless two-bay shelf
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

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from motion_planning.utils import Obstacle, Shelf
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    FixedZCostFunction,
    CompositeCostFunction,
    CostModeFactory,
    # NEW: make sure you added this class in your cost_functions.py
    ShelfSideWallRiskCost,
    MuJoCoEECollisionCost
)
from trajectory_optimization_demo import MultiTrajectoryDemo

# NEW: shelf geometry builder (add the helper we discussed to scene_builder.py)
from scene_builder_shelf import create_scene_with_elements


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

    def __init__(self, config: SearchConfiguration, args):
        
        self.args = args
        self.config = config
        # Shelf params cache (filled when we build the shelf scene file)
        self.shelf_box = None          # dict(x=(x0,x1), y=(y0,y1), z=(z0,z1))
        self.selected_bay = None       # (yL,yR)
        self._goal_override = None

        super().__init__()

        self.alpha_values = np.arange(config.alpha_start, config.alpha_end + config.alpha_step, config.alpha_step)
        self.colors = self._generate_plasma_colors(len(self.alpha_values))
        self.results = []
        self.experiment_dir = None
        self.trajectory_metadata = []
        

        

        # Setup experiment directory if saving is enabled
        if config.save_trajectories:
            self._setup_experiment_directory()
    
    # --------------------
    # Scenario definitions
    # --------------------
    def define_obstacles(self) -> List[Obstacle]:
        """Define obstacles for the Pareto search demo (balls task only)."""
        if getattr(self.args, "task", "shelf") == "shelf":
            # No standalone spherical obstacles for shelf task;
            # shelf geometry comes from the XML we build via MujocoSceneBuilder.
            return []
        return [
            Obstacle(center=np.array([0.35, 0.06, 0.2]), radius=0.04, safe_distance=0.04),
        ]

    def _shelf_center_goal_from_args(self) -> np.ndarray:
        x0, y0, z0 = self.args.shelf_origin
        depth = self.args.shelf_depth
        height = self.args.shelf_height
        wA, wB = self.args.bay_widths

        yL, yM, yR = y0, y0 + wA, y0 + wA + wB

        if self.args.bay == 'A':
            y_min, y_max = yL, yM
        elif self.args.bay == 'B':
            y_min, y_max = yM, yR
        else:  # 'auto' → center between both bays
            y_min, y_max = yL, yR

        gx = x0 + 0.5 * depth
        gy = 0.5 * (y_min + y_max)
        gz = z0 + 0.5 * height
        return np.array([gx, gy, gz])


    def define_target_position(self):
        if self._goal_override is not None:
            return self._goal_override
        if getattr(self.args, "task", "shelf") == "shelf":
            if self.args.bay == "auto":
                x0,y0,z0 = self.args.shelf_origin
                d,h = self.args.shelf_depth, self.args.shelf_height
                wA,wB = self.args.bay_widths
                return np.array([x0+0.5*d, y0+0.5*(wA+wB), z0+0.5*h])  # mid of both bays
            return self._shelf_center_goal_from_args()
        return np.array([0.65,0.00,0.2])


    def define_start_config(self) -> np.ndarray:
        """Define start configuration for the Pareto search demo"""
        return np.array([0.0, -0.5, 0.0, 2.5, 0.0, 1.0, -1.57])

    def get_max_trajectories(self) -> int:
        """Get maximum number of trajectories for Pareto search"""
        return len(self.alpha_values)

    def get_scene_filename(self) -> str:
        """Just a filename, not a path. The builder will put it in robot_models/kinova_gen3/."""
        if getattr(self.args, "task", "shelf") == "shelf":
            return "scene_shelf.xml"
        return "pareto_search_scene.xml"

    def define_target_position(self) -> np.ndarray:
        # use a runtime override if set (so we can plan for A and B)
        if self._goal_override is not None:
            return self._goal_override
        if getattr(self.args, "task", "shelf") == "shelf":
            return self._shelf_center_goal_from_args()   # default (uses args.bay)
        return np.array([0.65, 0.00, 0.2])

    def _goal_center_for_bay(self, which: str) -> np.ndarray:
        """Center of the chosen bay (A/B), mid-depth & mid-height, from args."""
        x0, y0, z0 = self.args.shelf_origin
        d  = self.args.shelf_depth
        h  = self.args.shelf_height
        wA, wB = self.args.bay_widths
        yL, yM, yR = y0, y0 + wA, y0 + wA + wB
        y_min, y_max = (yL, yM) if which.upper() == 'A' else (yM, yR)
        return np.array([x0 + 0.5*d, 0.5*(y_min+y_max), z0 + 0.5*h])
    

    def create_scene(self) -> str:
        """
        Build the MuJoCo scene. For 'shelf' task we inject a backless two-bay shelf
        (no back wall) and cache its geometry for the risk cost.
        """
        if getattr(self.args, "task", "shelf") == "shelf":
            # 1) Define the shelf
            shelves = [
                Shelf(
                    origin=tuple(self.args.shelf_origin),
                    depth=self.args.shelf_depth,
                    height=self.args.shelf_height,
                    bay_widths=tuple(self.args.bay_widths),
                    collidable=True
                )
            ]

            # 2) Build scene (builder writes to robot_models/kinova_gen3/<filename>)
            scene_path = create_scene_with_elements(
                obstacles=self.define_obstacles(),                 # empty for shelf task; OK
                target_position=self.define_target_position(),     # the green target marker
                max_traj=self.get_max_trajectories() * 50,
                filename=self.get_scene_filename(),                # "scene_shelf.xml"
                shelves=shelves,
                show_target=False
            )

            # 3) Cache shelf geometry for the risk cost
            sh = shelves[0]
            self.shelf_box = {'x': sh.x_bounds(), 'y': sh.y_bounds(), 'z': sh.z_bounds()}
            bayA, bayB = sh.bay_intervals()
            self.bay_intervals = {'A': bayA, 'B': bayB}
            self.selected_bay = bayA if self.args.bay == 'A' else (bayB if self.args.bay == 'B' else None)


            return scene_path

        # Fallback to the parent behavior for the 'balls' task
        return super().create_scene()


    def _shelf_center_goal_for_bay(self, which: str) -> np.ndarray:
        x0, y0, z0 = self.args.shelf_origin
        depth = self.args.shelf_depth
        height = self.args.shelf_height
        wA, wB = self.args.bay_widths
        yL, yM, yR = y0, y0 + wA, y0 + wA + wB
        if which.upper() == 'A':
            y_min, y_max = yL, yM
        else:
            y_min, y_max = yM, yR
        gx = x0 + 0.5 * depth
        gy = 0.5 * (y_min + y_max)
        gz = z0 + 0.5 * height
        return np.array([gx, gy, gz])

    def _plan_for_bay(self, bay: str, alpha: float, color, model, data, kinematics):
        # weights for this alpha
        w_len = alpha
        w_safe = 1.0 - alpha

        ...
        goal_xyz = self._shelf_center_goal_for_bay(bay)
        print(f"[{bay}] IK target xyz = {goal_xyz}")

        # --- force IK to use this target (works with the existing signature) ---
        old_target = getattr(self, "target_position", None)
        self.target_position = goal_xyz
        try:
            goal_q = self.solve_inverse_kinematics(kinematics, target_position=goal_xyz)  # uses self.target_position
        finally:
            self.target_position = old_target
        # -----------------------------------------------------------------------

        if goal_q is None:
            print(f"[{bay}] IK failed")
            return None
        ...

        # planner
        planner = self.create_planner(model, data)

        # length + side-wall risk for THIS bay
        length_cost = TrajectoryLengthCostFunction(kinematics_solver=kinematics, weight=1.0)

        bay_interval = self.bay_intervals[bay]
        safety_cost = ShelfSideWallRiskCost(
            kinematics_solver=kinematics,
            bay_interval=bay_interval,               # (yL, yR) for this bay
            shelf_x=self.shelf_box['x'],             # (xL, xR)
            shelf_z=self.shelf_box['z'],             # (zL, zR) floor & roof
            weight=1.0,
            decay_rate=self.args.risk_decay,
            aggregate=self.args.risk_agg,                         # keep worst-case
            margin=0.02,
            barrier_gain=2000.0,
            include_floor_roof=False,
            x_expand=0.0
        )

        # Use your CompositeCostFunction object for the SAME formulation
        comp = CompositeCostFunction(
            [length_cost, safety_cost],
            [w_len, w_safe],
            mode=self.config.cost_mode,
            rho=self.config.rho
        )
        planner.set_composite_cost_function(comp)  
        # comp = planner.setup_composite_cost([length_cost, safety_cost], [w_len, w_safe],
        #                                     formulation=self.config.cost_mode, rho=self.config.rho)

        # optimize
        traj, ok, meta = planner.plan(self.start_config, goal_q)
        if not ok or traj is None:
            return None

        traj_np = np.array(traj)
        # Evaluate with THE SAME composite object (the key part you wanted)
        J = comp.compute_cost(traj_np)

        # (optional) also log individual terms
        f_len = length_cost.compute_cost(traj_np)
        f_safe = safety_cost.compute_cost(traj_np)
        return {
            "bay": bay,
            "traj": traj,
            "color": color,
            "score": float(J),
            "f_len": float(f_len),
            "f_safe": float(f_safe),
            "meta": meta
        }


    def create_planner(self, model, data):
        """Create constrained trajectory optimization planner"""
        return ConstrainedTrajOptPlanner(
            model, data,
            n_waypoints=25,  # Reduced for faster iteration
            dt=0.1,
            max_velocity=1.0,
            max_acceleration=0.7,
            cost_mode='composite'
        )

    def setup_cost_functions(self, planner, kinematics) -> bool:
        """Setup cost functions for current alpha value (deferred to plan_single_trajectory)"""
        return True

    # --------------------
    # Utilities
    # --------------------
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
            'task': getattr(self.args, "task", "shelf"),
            'bay': getattr(self.args, "bay", "A"),
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
                          for obs in self.obstacles] if hasattr(self, 'obstacles') and self.obstacles else [],
            # Shelf params (if task == shelf)
            'shelf_origin': getattr(self.args, "shelf_origin", None),
            'shelf_depth': getattr(self.args, "shelf_depth", None),
            'shelf_height': getattr(self.args, "shelf_height", None),
            'bay_widths': getattr(self.args, "bay_widths", None),
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Experiment directory created: {self.experiment_dir}")

    def _save_trajectory(self, trajectory: List[np.ndarray], alpha: float,
                         length_cost: float, obstacle_or_risk_cost: float, color: np.ndarray,
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
            # keep key name 'obstacle_cost' for compatibility; this will hold 'risk' when task=shelf
            'obstacle_cost': obstacle_or_risk_cost,
            'color': color,
            'length_weight': alpha,
            'obstacle_weight': 1.0 - alpha,
            'trajectory_id': trajectory_id,
            'waypoint_count': len(trajectory),
            'timestamp': datetime.now().isoformat(),
            'task': getattr(self.args, "task", "shelf")
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
            'obstacle_cost': obstacle_or_risk_cost,  # risk when task=shelf
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

    # --------------------
    # Planning
    # --------------------
    # def plan_single_trajectory(self, alpha: float, color: np.ndarray, model, data, kinematics):
    #     """Plan a single trajectory for given alpha value"""
    #     print(f"Planning trajectory for α={alpha:.1f} ({self.config.cost_mode.upper()} mode)")

    #     # Calculate weights
    #     length_weight = alpha
    #     safety_weight = 1.0 - alpha

    #     # Setup IK
    #     goal_config = self.solve_inverse_kinematics(kinematics)
    #     if goal_config is None:
    #         return None

    #     # Create planner with composite cost mode
    #     planner = self.create_planner(model, data)

    #     try:
    #         # Create individual cost functions
    #         length_cost = TrajectoryLengthCostFunction(
    #             kinematics_solver=kinematics,
    #             weight=1.0,
    #             normalization_bounds=(0.0, 1.0)
    #         )

    #         task = getattr(self.args, "task", "shelf")
    #         if task == 'shelf':
    #             if self.shelf_box is None or self.selected_bay is None:
    #                 raise RuntimeError("Shelf parameters not initialized. Check get_scene_filename() call order.")
    #             safety_cost = ShelfSideWallRiskCost(
    #                 kinematics_solver=kinematics,
    #                 bay_interval=self.selected_bay,          # (yL, yR)
    #                 shelf_x=self.shelf_box['x'],             # (xL, xR)
    #                 shelf_z=self.shelf_box['z'],             # (zL, zR)  <-- floor & roof bounds
    #                 weight=1.0,
    #                 decay_rate=80.0,                         # steeper decay
    #                 aggregate='max',                         # worst waypoint drives the cost
    #                 margin=0.02,                             # 2 cm keep-out
    #                 barrier_gain=2000.0,
    #                 include_floor_roof=True,                 # <<< this is the key bit
    #                 x_expand=0.0                             # optional tolerance in x
    #             )
    #             # ONLY end-effector vs shelf
    #             # safety_cost = MuJoCoEECollisionCost(
    #             #     model, data,
    #             #     ee_geom_prefixes=("ee_",),           # or ee_body_names=("tool_tip",)
    #             #     shelf_geom_prefixes=("shelf_",),
    #             #     qpos_arm_idx=[0,1,2,3,4,5,6],        # mapping 7-DOF arm → qpos slots
    #             #     rest_qpos=data.qpos.copy(),          # keeps fingers/base as-is
    #             #     margin=0.02,
    #             #     weight=1.0,
    #             #     aggregate="max"
    #             # )
    #         else:
    #             safety_cost = ObstacleAvoidanceCostFunction(
    #                 kinematics_solver=kinematics,
    #                 obstacles=self.obstacles,
    #                 weight=1.0,
    #                 normalization_bounds=(0.0, 1.0),
    #                 decay_rate=15.0,
    #                 bias=-0.08,
    #                 aggregate="avg"
    #             )

    #         # Set up composite cost function
    #         cost_functions = [length_cost, safety_cost]
    #         weights = [length_weight, safety_weight]

    #         composite_cost = planner.setup_composite_cost(
    #             cost_functions=cost_functions,
    #             weights=weights,
    #             formulation=self.config.cost_mode,
    #             rho=self.config.rho
    #         )

    #         # Plan trajectory
    #         trajectory, success, optimization_metadata = planner.plan(self.start_config, goal_config)

    #         if success:
    #             self.add_trajectory(trajectory, color)

    #             trajectory_np = np.array(trajectory)  # shape: (N_waypoints, DOF)

    #             f_length = length_cost.compute_cost(trajectory_np)
    #             f_safety = safety_cost.compute_cost(trajectory_np)

    #             print(f"Length cost for α={alpha:.1f}: {f_length:.4f}")
    #             if task == 'shelf':
    #                 print(f"Side-wall RISK (avg) for α={alpha:.1f}: {f_safety:.4f}")
    #             else:
    #                 print(f"Closeness (obstacle) for α={alpha:.1f}: {f_safety:.4f}")

    #             self.results.append((f_length, f_safety, alpha))

    #             # Save trajectory if enabled (with optimization metadata)
    #             self._save_trajectory(trajectory, alpha, f_length, f_safety, color, optimization_metadata)

    #             print(f"α={alpha:.1f}: Success")
    #             return trajectory
    #         else:
    #             print(f"α={alpha:.1f}: Failed")
    #             return None

    #     except Exception as e:
    #         print(f"α={alpha:.1f}: Error - {e}")
    #         return None

    def plan_single_trajectory(self, alpha: float, color, model, data, kinematics):
        print(f"Planning for α={alpha:.1f} (choose bay A vs B)")
        candA = self._plan_for_bay('A', alpha, color, model, data, kinematics)
        candB = self._plan_for_bay('B', alpha, color, model, data, kinematics)

        if candA is None and candB is None:
            print(f"α={alpha:.1f}: both bays failed")
            return None

        best = None
        if candA is not None and candB is not None:
            best = candA if candA["score"] <= candB["score"] else candB
        else:
            best = candA if candA is not None else candB

        # store & log
        self.add_trajectory(best["traj"], color)
        traj_np = np.array(best["traj"])
        self.results.append((best["f_len"], best["f_safe"], alpha))
        self._save_trajectory(best["traj"], alpha, best["f_len"], best["f_safe"], color, best["meta"])
        print(f"α={alpha:.1f}: chose bay {best['bay']} | J={best['score']:.4f} | L={best['f_len']:.4f} | Risk={best['f_safe']:.4f}")
        return best["traj"]


    # --------------------
    # Orchestration
    # --------------------
    def run_pareto_search(self, model, data, kinematics):
        """Run the complete Pareto search"""
        print("Starting Linear Weight Search")
        print(f"Task: {getattr(self.args, 'task', 'shelf')}")
        print(f"Cost formulation: {self.config.cost_mode.upper()}")
        print(f"Alpha range: [{self.config.alpha_start:.1f}, {self.config.alpha_end:.1f}] step {self.config.alpha_step:.1f}")

        successful_count = 0

        for i, alpha in enumerate(self.alpha_values):
            color = self.colors[i]
            trajectory = self.plan_single_trajectory(float(alpha), color, model, data, kinematics)
            if trajectory is not None:
                successful_count += 1

        print(f"Search complete: {successful_count}/{len(self.alpha_values)} successful trajectories")

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

    def save_results_to_csv(self, output_dir="src/pareto_data_and_results", filename="Shelving_test.csv"):
        if not self.results:
            print("No results to save.")
            return
        os.makedirs(output_dir, exist_ok=True)

        full_path = os.path.join(output_dir, filename)

        with open(full_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Keep your old header for compatibility; "closeness" will be "risk" for shelf task
            writer.writerow(["length", "closeness", "alpha"])
            writer.writerows(self.results)

        print(f"Results saved to {full_path}")

    def execute_planning_loop(self, model, data, kinematics, viewer_handle):
        """Execute Pareto search and visualize results"""
        # Run the search
        self.run_pareto_search(model, data, kinematics)

        # Plot trade-off between length and safety (saved CSV)
        self.save_results_to_csv()  # Use default output_dir and filename

        # Visualize all trajectories
        if self.trajectories:
            self.execute_trajectory(viewer_handle, model, data, kinematics, None)
        else:
            print("No successful trajectories to visualize")


def parse_arguments():
    """Parse command line arguments for cost mode and search parameters"""
    parser = argparse.ArgumentParser(description='Linear Weight Search for Trajectory Optimization Pareto Analysis')

    parser.add_argument('--task', choices=['balls', 'shelf'], default='shelf',
                        help='Task type: "balls" uses obstacle avoidance; "shelf" uses side-wall risk (default: shelf)')
    parser.add_argument('--bay', choices=['A', 'B', 'auto'], default='auto',
                        help='bay to target, or "auto" to let the planner choose per-α')

    # Shelf geometry params (only used if task=shelf)
    parser.add_argument('--shelf-origin', nargs=3, type=float, default=[0.58, -0.10, 0.12],
                        help='Front-left-bottom corner of shelf: x y z (default: 0.58 -0.10 0.12)')
    parser.add_argument('--shelf-depth', type=float, default=0.30,
                        help='Shelf depth along +x (default: 0.30)')
    parser.add_argument('--shelf-height', type=float, default=0.22,
                        help='Shelf inner height along +z (default: 0.22)')
    parser.add_argument('--bay-widths', nargs=2, type=float, default=[0.08, 0.16],
                        help='Bay widths (narrow,wide) along +y (default: 0.08 0.16)')

    # Risk cost params
    parser.add_argument('--risk-decay', type=float, default=25.0,
                        help='Exponential decay rate for side-wall risk (default: 25.0)')
    parser.add_argument('--risk-agg', choices=['avg', 'sum', 'min', 'max'], default='avg',
                        help='Aggregate for risk across waypoints (default: avg)')

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

    # Data saving
    parser.add_argument('--csv-file', type=str, default='tradeoff_data.csv',
                        help='Name of the CSV file to save results (default: tradeoff_data.csv)')
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
    print(f"Task: {args.task}")
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
    demo = ParetoSearchDemo(config, args)
    demo.run_demo()
