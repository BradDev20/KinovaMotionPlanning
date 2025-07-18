#!/usr/bin/env python3
"""
Linear Weight Search for Trajectory Optimization Pareto Analysis

This script performs a linear search over weights for trajectory length and obstacle avoidance,
visualizing all resulting trajectories simultaneously with a plasma colormap.

Supports both linear weighted sum and weighted maximum cost formulations.

Key features:
- Linear search: alpha ∈ [0, 1] discretized every 0.1
- Weight mapping: alpha → trajectory_length_weight, (1-alpha) → obstacle_avoidance_weight  
- Simultaneous visualization of all trajectories with plasma colormap
- Support for both 'sum' and 'max' cost formulations
- Clean, extensible architecture for future modifications
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import sys
import os
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Try to import matplotlib for colormap, fallback if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from motion_planning.utils import Obstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    FixedZCostFunction,
    CompositeCostFunction,
    CostModeFactory
)


@dataclass
class SearchConfiguration:
    """Configuration for the linear weight search"""
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_step: float = 1.0
    n_waypoints: int = 50
    dt: float = 0.1
    max_velocity: float = 1.0
    max_acceleration: float = 0.7
    cost_mode: str = 'sum'  # 'sum' or 'max'
    rho: float = 0.01  # Tie-breaking parameter for max mode
    target_position: Optional[np.ndarray] = None
    start_config: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.target_position is None:
            self.target_position = np.array([-0.7, 0.1, 0.529])
        if self.start_config is None:
            self.start_config = np.array([0.0, 0.5, 0.0, -2.5, 0.0, 0.45, 1.57])


@dataclass 
class TrajectoryResult:
    """Container for a single trajectory optimization result"""
    alpha: float
    trajectory: Optional[List[np.ndarray]]
    success: bool
    planning_time: float
    cost: float
    length_weight: float
    obstacle_weight: float
    color: np.ndarray
    cost_mode: str


class ParetoSearchVisualizer:
    """Main class for linear weight search and visualization"""
    
    def __init__(self, config: SearchConfiguration, obstacles: List[Obstacle]):
        self.config = config
        self.obstacles = obstacles
        self.results: List[TrajectoryResult] = []
        self.model = None
        self.data = None
        self.kinematics = None
        
        # Generate plasma colormap for alpha values
        self.alpha_values = np.arange(config.alpha_start, config.alpha_end + config.alpha_step, config.alpha_step)
        self.colors = self._generate_plasma_colors(len(self.alpha_values))
        
    def _generate_plasma_colors(self, n_colors: int) -> np.ndarray:
        """Generate plasma colormap colors for the trajectories"""
        if HAS_MATPLOTLIB:
            # Use matplotlib's plasma colormap
            plasma = plt.cm.plasma
            color_indices = np.linspace(0, 1, n_colors)
            colors = []
            for idx in color_indices:
                rgba = plasma(idx)
                colors.append([rgba[0], rgba[1], rgba[2], 0.8])  # RGB + alpha
            return np.array(colors)
        else:
            # Fallback: simple interpolation from purple to yellow
            colors = []
            for i in range(n_colors):
                t = i / (n_colors - 1) if n_colors > 1 else 0
                # Purple to yellow interpolation
                r = 0.5 + 0.5 * t  # 0.5 -> 1.0
                g = 0.0 + 1.0 * t  # 0.0 -> 1.0  
                b = 1.0 - 1.0 * t  # 1.0 -> 0.0
                colors.append([r, g, b, 0.8])
            return np.array(colors)
    
    def setup_scene(self) -> str:
        """Create MuJoCo scene with obstacles and trajectory trace dots"""
        base_scene_path = 'robot_models/kinova_gen3/scene.xml'
        
        if not os.path.exists(base_scene_path):
            base_scene_path = 'robot_models/kinova_gen3/gen3.xml'
        
        with open(base_scene_path, 'r') as f:
            scene_content = f.read()
        
        # Generate XML for obstacles
        obstacle_xml_parts = []
        for i, obstacle in enumerate(self.obstacles):
            obstacle_xml = f'''
    <!-- Obstacle {i+1} visualization -->
    <body name="obstacle_{i+1}" pos="{obstacle.center[0]:.3f} {obstacle.center[1]:.3f} {obstacle.center[2]:.3f}">
        <geom name="obstacle_{i+1}_geom" type="sphere" size="{obstacle.radius}" 
              rgba="1.0 0.2 0.2 0.6" material="" contype="0" conaffinity="0"/>
        <site name="obstacle_{i+1}_center" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
    </body>
'''
            obstacle_xml_parts.append(obstacle_xml)
        
        # Add target position marker
        target_pos = self.config.target_position
        assert target_pos is not None, "Target position must be set"
        target_xml = f'''
    <!-- Target position marker -->
    <body name="target_marker" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}">
        <geom name="target_geom" type="sphere" size="0.03" 
              rgba="0.0 0.8 0.0 0.8" material="" contype="0" conaffinity="0"/>
        <site name="target_center" pos="0 0 0" size="0.005" rgba="0 1 0 1"/>
    </body>
'''
        
        # Generate trajectory trace dots (enough for all trajectories)
        max_dots_needed = len(self.alpha_values) * 100  # Conservative estimate
        trace_xml_parts = []
        for i in range(max_dots_needed):
            trace_xml = f'''
    <body name="trace_dot_{i}" pos="0 0 -10">
        <geom name="trace_dot_{i}_geom" type="sphere" size="0.008" 
              rgba="1.0 1.0 1.0 0.8" material="" contype="0" conaffinity="0"/>
    </body>'''
            trace_xml_parts.append(trace_xml)
        
        # Combine all XML
        all_xml = ''.join(obstacle_xml_parts) + target_xml + ''.join(trace_xml_parts)
        
        # Insert before closing worldbody tag
        insert_pos = scene_content.rfind('</worldbody>')
        if insert_pos == -1:
            insert_pos = scene_content.rfind('</mujoco>')
            if insert_pos == -1:
                raise ValueError("Could not find insertion point in XML")
        
        modified_content = (scene_content[:insert_pos] + all_xml + scene_content[insert_pos:])
        
        # Save to new file
        output_path = 'robot_models/kinova_gen3/pareto_search_scene.xml'
        with open(output_path, 'w') as f:
            f.write(modified_content)
        
        return output_path
    
    def initialize_model(self, scene_path: str):
        """Initialize MuJoCo model and kinematics solver"""
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.kinematics = KinematicsSolver(scene_path)
        
    def plan_single_trajectory(self, alpha: float, color: np.ndarray) -> TrajectoryResult:
        """Plan a single trajectory for given alpha value using composite cost system"""
        print(f"\n🔄 Planning trajectory for α={alpha:.1f} ({self.config.cost_mode.upper()} mode)")
        
        # Ensure model and kinematics are initialized
        assert self.kinematics is not None, "Kinematics solver must be initialized"
        assert self.config.target_position is not None, "Target position must be set"
        assert self.config.start_config is not None, "Start config must be set"
        
        # Calculate weights
        length_weight = alpha
        obstacle_weight = 1.0 - alpha
        z_weight = 0.5  # Small fixed Z weight
        print(f"   Weights: Length={length_weight:.1f}, Obstacle={obstacle_weight:.1f}, Z={z_weight:.2f}")
        
        # Setup IK to find goal configuration
        goal_config, ik_success = self.kinematics.inverse_kinematics(
            self.config.target_position,
            initial_guess=self.config.start_config,
            tolerance=0.001,
            max_iterations=2000
        )
        
        if not ik_success:
            print(f"  ❌ IK failed for α={alpha:.1f}")
            return TrajectoryResult(
                alpha=alpha, trajectory=None, success=False, planning_time=0.0,
                cost=float('inf'), length_weight=length_weight, obstacle_weight=obstacle_weight,
                color=color, cost_mode=self.config.cost_mode
            )
        
        # Create planner with composite cost mode
        planner = ConstrainedTrajOptPlanner(
            self.model, self.data, 
            n_waypoints=self.config.n_waypoints,
            dt=self.config.dt,
            max_velocity=self.config.max_velocity,
            max_acceleration=self.config.max_acceleration,
            # max_acceleration=10.0,
            cost_mode='composite'  # Use new composite cost system
        )
        
        try:
            # Create individual cost functions
            length_cost = TrajectoryLengthCostFunction(
                weight=1.0,  # Weight will be handled by composite function
                normalization_bounds=(1.0, 2.0)
            )

            safety_cost = ObstacleAvoidanceCostFunction(
                kinematics_solver=self.kinematics,
                obstacles=self.obstacles,
                weight=4.0,  # Weight will be handled by composite function
                normalization_bounds=(0.0, 0.8),
                decay_rate=5.0  # Exponential decay rate - matches pareto.py
            )

            fixed_z_cost = FixedZCostFunction(
                kinematics_solver=self.kinematics,
                target_z=0.529,
                weight=1.0  # Weight will be handled by composite function
            )

            # Set up composite cost function with specified mode
            cost_functions = [length_cost, safety_cost]
            weights = [length_weight, obstacle_weight]
            
            composite_cost = planner.setup_composite_cost(
                cost_functions=cost_functions,
                weights=weights,
                formulation=self.config.cost_mode,
                rho=self.config.rho
            )

        except Exception as e:
            print(f"  ❌ Error setting up cost functions: {e}")
            return TrajectoryResult(
                alpha=alpha, trajectory=None, success=False, planning_time=0.0,
                cost=float('inf'), length_weight=length_weight, obstacle_weight=obstacle_weight,
                color=color, cost_mode=self.config.cost_mode
            )
        
        # Plan trajectory
        print(f"   Starting trajectory planning...")
        start_time = time.time()
        try:
            trajectory, success = planner.plan(self.config.start_config, goal_config)
            planning_time = time.time() - start_time
            print(f"   Planning completed: success={success}, time={planning_time:.2f}s")
        except Exception as e:
            planning_time = time.time() - start_time
            print(f"   ❌ Planning failed with exception: {e}")
            trajectory, success = None, False
        
        # Calculate final cost (for analysis) - use the composite cost
        final_cost = 0.0
        if success and trajectory:
            try:
                final_cost = composite_cost.compute_cost(np.array(trajectory), self.config.dt)
            except:
                final_cost = float('inf')
        
        print(f"  ✅ α={alpha:.1f}: {'Success' if success else 'Failed'}, "
              f"cost={final_cost:.3f}, time={planning_time:.2f}s")
        
        return TrajectoryResult(
            alpha=alpha, trajectory=trajectory, success=success, planning_time=planning_time,
            cost=final_cost, length_weight=length_weight, obstacle_weight=obstacle_weight,
            color=color, cost_mode=self.config.cost_mode
        )
    
    def run_weight_search(self):
        """Run the complete linear weight search"""
        print("🔍 Starting Linear Weight Search")
        print(f"Cost formulation: {self.config.cost_mode.upper()} ({'Linear Weighted Sum' if self.config.cost_mode == 'sum' else 'Weighted Maximum with Tie-breaking'})")
        if self.config.cost_mode == 'max':
            print(f"Tie-breaking ρ: {self.config.rho}")
        print(f"Alpha range: [{self.config.alpha_start:.1f}, {self.config.alpha_end:.1f}] step {self.config.alpha_step:.1f}")
        print(f"Number of trajectories: {len(self.alpha_values)}")
        print("🚨 Using EXPONENTIAL DECAY for obstacle avoidance (decay_rate=5.0)")
        print("=" * 60)
        
        # Plan all trajectories
        for i, alpha in enumerate(self.alpha_values):
            print(f"\n📍 Starting trajectory {i+1}/{len(self.alpha_values)}: α={alpha:.1f}")
            color = self.colors[i]
            try:
                result = self.plan_single_trajectory(float(alpha), color)
                self.results.append(result)
                print(f"📍 Completed trajectory {i+1}/{len(self.alpha_values)}: α={alpha:.1f}, success={result.success}")
            except Exception as e:
                print(f"❌ Critical error for α={alpha:.1f}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next alpha value
                continue
        
        # Print summary
        successful_results = [r for r in self.results if r.success]
        print("\n" + "=" * 60)
        print("🎯 SEARCH COMPLETE")
        print(f"Cost formulation: {self.config.cost_mode.upper()}")
        print(f"Successful trajectories: {len(successful_results)}/{len(self.results)}")
        
        if successful_results:
            avg_time = np.mean([r.planning_time for r in successful_results])
            print(f"Average planning time: {avg_time:.2f}s")
            
            costs = [r.cost for r in successful_results]
            print(f"Cost range: [{min(costs):.3f}, {max(costs):.3f}]")
    
    def visualize_trajectories(self):
        """Visualize all successful trajectories simultaneously"""
        if not any(r.success for r in self.results):
            print("❌ No successful trajectories to visualize")
            return
        
        # Ensure model, data, and kinematics are initialized
        assert self.model is not None, "Model must be initialized"
        assert self.data is not None, "Data must be initialized"
        assert self.kinematics is not None, "Kinematics solver must be initialized"
        assert self.config.start_config is not None, "Start config must be set"
        
        print(f"\n🎨 Starting trajectory visualization ({self.config.cost_mode.upper()} mode)...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer_handle:
            # Set robot to home position
            self.data.qpos[:7] = self.config.start_config
            self.data.ctrl[:7] = self.config.start_config
            mujoco.mj_forward(self.model, self.data)
            viewer_handle.sync()
            
            print("Rendering all trajectories...")
            
            # Render each successful trajectory
            dot_offset = 0
            for result in self.results:
                if not result.success or result.trajectory is None:
                    continue
                    
                # Convert trajectory to end-effector positions
                ee_positions = []
                for waypoint in result.trajectory:
                    ee_pos, _ = self.kinematics.forward_kinematics(waypoint)
                    ee_positions.append(ee_pos)
                
                # Render trajectory trace with appropriate color
                self._render_trajectory_trace(ee_positions, result.color, dot_offset)
                dot_offset += len(ee_positions)
            
            # Update viewer
            mujoco.mj_forward(self.model, self.data)
            viewer_handle.sync()
            
            print(f"\n🌈 All trajectories rendered ({self.config.cost_mode.upper()} mode)!")
            self._print_legend()
            self._interactive_viewer_controls(viewer_handle)
    
    def _render_trajectory_trace(self, ee_positions: List[np.ndarray], color: np.ndarray, dot_offset: int):
        """Render a single trajectory trace with given color"""
        for i, pos in enumerate(ee_positions):
            dot_id = dot_offset + i
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{dot_id}")
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"trace_dot_{dot_id}_geom")
                if body_id >= 0 and geom_id >= 0:
                    self.model.body_pos[body_id] = pos
                    self.model.geom_rgba[geom_id] = color
            except:
                pass  # Skip if dot doesn't exist
    
    def _print_legend(self):
        """Print color legend for trajectories"""
        print(f"\n🏷️  TRAJECTORY LEGEND ({self.config.cost_mode.upper()} mode):")
        print("=" * 50)
        for i, result in enumerate(self.results):
            if result.success:
                print(f"α={result.alpha:.1f}: length_weight={result.length_weight:.1f}, "
                      f"obstacle_weight={result.obstacle_weight:.1f}, cost={result.cost:.3f}")
        print("📊 Color scale: Purple (α=0, obstacle focus) → Yellow (α=1, length focus)")
        if self.config.cost_mode == 'max':
            print(f"🔧 Weighted Maximum with tie-breaking ρ={self.config.rho}")
        else:
            print("🔧 Linear Weighted Sum")
    
    def _interactive_viewer_controls(self, viewer_handle):
        """Interactive controls for the viewer"""
        print("\n" + "=" * 60)
        print("INTERACTIVE CONTROLS:")
        print("  [s] - Show statistics")
        print("  [l] - Show legend")
        print("  [h] - Show home position")
        print("  [q] - Quit")
        print("=" * 60)
        
        while True:
            try:
                choice = input("Enter your choice: ").strip().lower()
                
                if choice == 's':
                    self._print_statistics()
                elif choice == 'l':
                    self._print_legend()
                elif choice == 'h':
                    self.data.qpos[:7] = self.config.start_config
                    self.data.ctrl[:7] = self.config.start_config
                    mujoco.mj_forward(self.model, self.data)
                    viewer_handle.sync()
                    print("Robot moved to home position!")
                elif choice == 'q':
                    print("\n👋 Exiting visualization...")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
                # Keep simulation running
                for _ in range(10):
                    mujoco.mj_step(self.model, self.data)
                    viewer_handle.sync()
                    time.sleep(0.01)
                    
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Exiting visualization...")
                break
    
    def _print_statistics(self):
        """Print detailed statistics about the search results"""
        successful_results = [r for r in self.results if r.success]
        
        print(f"\n📊 DETAILED STATISTICS ({self.config.cost_mode.upper()} mode):")
        print("=" * 60)
        print(f"Cost formulation: {self.config.cost_mode.upper()}")
        if self.config.cost_mode == 'max':
            print(f"Tie-breaking ρ: {self.config.rho}")
        print(f"Total trajectories planned: {len(self.results)}")
        print(f"Successful trajectories: {len(successful_results)}")
        print(f"Success rate: {len(successful_results)/len(self.results)*100:.1f}%")
        
        if successful_results:
            costs = [r.cost for r in successful_results]
            times = [r.planning_time for r in successful_results]
            
            print(f"\nCost statistics:")
            print(f"  Min cost: {min(costs):.3f} (α={successful_results[np.argmin(costs)].alpha:.1f})")
            print(f"  Max cost: {max(costs):.3f} (α={successful_results[np.argmax(costs)].alpha:.1f})")
            print(f"  Mean cost: {np.mean(costs):.3f}")
            
            print(f"\nPlanning time statistics:")
            print(f"  Min time: {min(times):.2f}s")
            print(f"  Max time: {max(times):.2f}s") 
            print(f"  Mean time: {np.mean(times):.2f}s")


def parse_arguments():
    """Parse command line arguments for cost mode and search parameters"""
    parser = argparse.ArgumentParser(
        description='Linear Weight Search for Trajectory Optimization Pareto Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pareto_search.py --cost-mode sum                    # Linear weighted sum search
  python pareto_search.py --cost-mode max --rho 0.005       # Weighted maximum search
  python pareto_search.py --alpha-step 0.05                 # Finer search resolution
        """
    )
    
    parser.add_argument('--cost-mode', 
                       choices=['sum', 'max'], 
                       default='sum',
                       help='Cost function formulation: "sum" for linear weighted sum, "max" for weighted maximum with tie-breaking (default: sum)')
    
    parser.add_argument('--rho',
                       type=float,
                       default=0.01,
                       help='Tie-breaking parameter for max mode (typically 0.001-0.1, default: 0.01)')
    
    parser.add_argument('--alpha-start',
                       type=float,
                       default=0.0,
                       help='Start value for alpha parameter (default: 0.0)')
    
    parser.add_argument('--alpha-end',
                       type=float,
                       default=1.0,
                       help='End value for alpha parameter (default: 1.0)')
    
    parser.add_argument('--alpha-step',
                       type=float,
                       default=0.1,
                       help='Step size for alpha parameter (default: 0.1)')
    
    return parser.parse_args()


def main():
    """Main function to run the linear weight search with configurable cost modes"""
    args = parse_arguments()
    
    print("🤖 Linear Weight Search for Trajectory Optimization")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Cost Mode: {args.cost_mode.upper()} ({'Linear Weighted Sum' if args.cost_mode == 'sum' else 'Weighted Maximum with Tie-breaking'})")
    if args.cost_mode == 'max':
        print(f"  Tie-breaking ρ: {args.rho}")
    print(f"  Alpha range: [{args.alpha_start:.1f}, {args.alpha_end:.1f}] step {args.alpha_step:.1f}")
    
    # Define obstacles (exactly matching pareto.py)
    obstacles = [
        # Obstacle(center=np.array([-0.6, -0.05, 0.529]), radius=0.07, safe_distance=0.05),
        Obstacle(center=np.array([-0.55, 0.05, 0.629]), radius=0.05, safe_distance=0.05),
        # Obstacle(center=np.array([-0.5, 0.15, 0.529]), radius=0.07, safe_distance=0.05),
        Obstacle(center=np.array([-0.55, 0.05, 0.429]), radius=0.05, safe_distance=0.05),
    ]
    
    print("=" * 70)
    print(f"Demonstrating search with {len(obstacles)} obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  {i+1}. Center: ({obs.center[0]:.2f}, {obs.center[1]:.2f}, {obs.center[2]:.2f}), "
              f"Radius: {obs.radius:.3f}m, Safety: {obs.safe_distance:.3f}m")
    print("=" * 70)
    
    # Create configuration
    config = SearchConfiguration(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_step=args.alpha_step,
        cost_mode=args.cost_mode,
        rho=args.rho
    )
    
    # Create and run visualizer
    visualizer = ParetoSearchVisualizer(config, obstacles)
    
    try:
        # Setup scene and model
        scene_path = visualizer.setup_scene()
        print(f"✓ Created scene: {scene_path}")
        
        visualizer.initialize_model(scene_path)
        print("✓ Model initialized")
        
        # Run weight search
        visualizer.run_weight_search()
        
        # Visualize results
        visualizer.visualize_trajectories()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 