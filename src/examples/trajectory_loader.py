#!/usr/bin/env python3
"""
Trajectory Loader and Visualizer for Pareto Search Results

This script allows you to load and visualize individual trajectories from 
Pareto search experiments without re-running the optimization.
"""

import sys
import os
import numpy as np
import argparse
import pickle
import json
from typing import List, Dict, Optional, Tuple
import mujoco

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.utils import Obstacle, PillarObstacle
from scene_builder import create_standard_scene
from trajectory_visualizer import TrajectoryVisualizationManager


class TrajectoryLoader:
    """Loads and manages saved trajectories from Pareto search experiments"""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the trajectory loader.
        
        Args:
            experiment_dir: Path to the experiment directory containing saved trajectories
        """
        self.experiment_dir = experiment_dir
        self.config = None
        self.metadata = None
        self.available_trajectories = []
        
        self._load_experiment_data()
        
    def _load_experiment_data(self):
        """Load experiment configuration and trajectory metadata"""
        # Load experiment configuration
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Experiment config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load trajectory metadata
        metadata_path = os.path.join(self.experiment_dir, "trajectory_metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Warning: No trajectory metadata found at {metadata_path}")
            print("This usually means no trajectories were successfully optimized and saved.")
            self.metadata = []
        else:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
        # Build list of available trajectories
        for traj_info in self.metadata:
            traj_path = os.path.join(self.experiment_dir, traj_info['filename'])
            if os.path.exists(traj_path):
                self.available_trajectories.append(traj_info)
                
        print(f"Loaded experiment: {self.config['experiment_name']}")
        print(f"Found {len(self.available_trajectories)} saved trajectories")
        
        if len(self.available_trajectories) == 0:
            print("No trajectories available for visualization.")
        
    def list_trajectories(self) -> None:
        """Print a list of available trajectories with their properties"""
        print(f"\nAvailable trajectories in {self.experiment_dir}:")
        
        if len(self.available_trajectories) == 0:
            print("No trajectories found in this experiment.")
            print("This could mean:")
            print("  1. The Pareto search failed before optimizing any trajectories")
            print("  2. All trajectory optimizations failed")
            print("  3. Trajectory saving was disabled")
            return
            
        print("=" * 80)
        print(f"{'ID':<15} {'Alpha':<8} {'Length':<10} {'Obstacle':<10} {'Waypoints':<10}")
        print("-" * 80)
        
        for traj_info in self.available_trajectories:
            print(f"{traj_info['trajectory_id']:<15} "
                  f"{traj_info['alpha']:<8.3f} "
                  f"{traj_info['length_cost']:<10.4f} "
                  f"{traj_info['obstacle_cost']:<10.4f} "
                  f"{traj_info['waypoint_count']:<10}")
                  
    def load_trajectory(self, trajectory_id: str) -> Optional[Dict]:
        """
        Load a specific trajectory by ID.
        
        Args:
            trajectory_id: ID of the trajectory to load
            
        Returns:
            Dictionary containing trajectory data or None if not found
        """
        # Find trajectory metadata
        traj_info = None
        for info in self.available_trajectories:
            if info['trajectory_id'] == trajectory_id:
                traj_info = info
                break
                
        if traj_info is None:
            print(f"Trajectory {trajectory_id} not found")
            return None
            
        # Load trajectory data
        traj_path = os.path.join(self.experiment_dir, traj_info['filename'])
        with open(traj_path, 'rb') as f:
            trajectory_data = pickle.load(f)
            
        print(f"Loaded trajectory: {trajectory_id}")
        print(f"  Alpha: {trajectory_data['alpha']:.3f}")
        print(f"  Length cost: {trajectory_data['length_cost']:.4f}")
        print(f"  Obstacle cost: {trajectory_data['obstacle_cost']:.4f}")
        print(f"  Waypoints: {trajectory_data['waypoint_count']}")
        
        return trajectory_data
        
    def get_experiment_obstacles(self) -> List[Obstacle]:
        """Recreate obstacles from experiment configuration"""
        obstacles = []
        for obs_data in self.config['obstacles']:
            if 'height' in obs_data:
                # This is a pillar obstacle
                obstacles.append(PillarObstacle(
                    center=np.array(obs_data['center']),
                    radius=obs_data['radius'],
                    height=obs_data['height'],
                    safe_distance=obs_data['safe_distance']
                ))
            else:
                # Regular spherical obstacle
                obstacles.append(Obstacle(
                    center=np.array(obs_data['center']),
                    radius=obs_data['radius'],
                    safe_distance=obs_data['safe_distance']
                ))
        return obstacles
        
    def get_experiment_target(self) -> np.ndarray:
        """Get target position from experiment configuration"""
        return np.array(self.config['target_position'])
        
    def get_experiment_start_config(self) -> np.ndarray:
        """Get start configuration from experiment configuration"""
        return np.array(self.config['start_config'])


class TrajectoryVisualizationDemo:
    """Demo class for visualizing loaded trajectories"""
    
    def __init__(self, loader: TrajectoryLoader):
        """
        Initialize the visualization demo.
        
        Args:
            loader: TrajectoryLoader instance with loaded experiment data
        """
        self.loader = loader
        self.obstacles = loader.get_experiment_obstacles()
        self.target_position = loader.get_experiment_target()
        self.start_config = loader.get_experiment_start_config()
        
    def create_scene(self) -> str:
        """Create MuJoCo scene with experiment obstacles and target"""
        scene_path = create_standard_scene(
            obstacles=self.obstacles,
            target_position=self.target_position,
            trace_dot_count=500,
            output_filename="trajectory_loader_scene.xml"
        )
        # create_standard_scene returns a path relative to the robot_models directory
        # but we need the full path from the current working directory
        return scene_path
        
    def visualize_trajectory(self, trajectory_data: Dict) -> None:
        """
        Visualize a loaded trajectory in MuJoCo.
        
        Args:
            trajectory_data: Dictionary containing trajectory and metadata
        """
        print(f"Visualizing trajectory: {trajectory_data['trajectory_id']}")
        
        try:
            # Create scene and load model
            model_path = self.create_scene()
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)
            kinematics = KinematicsSolver(model_path)
            
            # Convert trajectory back to list of arrays
            trajectory = [np.array(waypoint) for waypoint in trajectory_data['trajectory']]
            
            # Launch viewer and visualize
            print("Launching viewer...")
            with mujoco.viewer.launch_passive(model, data) as viewer_handle:
                # Setup initial position
                data.qpos[:7] = self.start_config
                data.ctrl[:7] = self.start_config
                mujoco.mj_forward(model, data)
                viewer_handle.sync()
                
                # Create trajectory visualizer
                visualizer = TrajectoryVisualizationManager(model, data, viewer_handle, kinematics)
                visualizer.set_target_position(self.target_position)
                
                # Execute trajectory with original color
                trajectory_color = np.array(trajectory_data['color'])
                stats = visualizer.execute_trajectory(
                    trajectory, 
                    trace_color=trajectory_color,
                    verbose=True
                )
                
                # Custom commands for this demo
                custom_commands = {
                    'i': ("Show trajectory info", lambda: self._print_trajectory_info(trajectory_data)),
                    's': ("Show statistics", lambda: self._print_statistics(stats))
                }
                
                # Run interactive session
                print("Trajectory visualization complete!")
                while True:
                    should_quit = visualizer.interactive_controls(
                        self.start_config, 
                        custom_commands, 
                        show_menu=True
                    )
                    if should_quit:
                        break
                        
        except FileNotFoundError as e:
            print(f"Error: Could not find required files")
            print(f"   {e}")
            print("   Make sure you're running from the project root directory")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            
    def _print_trajectory_info(self, trajectory_data: Dict):
        """Print detailed trajectory information"""
        print("\n" + "="*60)
        print("TRAJECTORY INFORMATION")
        print("="*60)
        print(f"Trajectory ID: {trajectory_data['trajectory_id']}")
        print(f"Alpha (length weight): {trajectory_data['alpha']:.3f}")
        print(f"Length weight: {trajectory_data['length_weight']:.3f}")
        print(f"Obstacle weight: {trajectory_data['obstacle_weight']:.3f}")
        print(f"Length cost: {trajectory_data['length_cost']:.4f}")
        print(f"Obstacle cost: {trajectory_data['obstacle_cost']:.4f}")
        print(f"Waypoint count: {trajectory_data['waypoint_count']}")
        print(f"Saved at: {trajectory_data['timestamp']}")
        print("="*60)
        
    def _print_statistics(self, stats: Dict):
        """Print execution statistics"""
        print("\n" + "="*60)
        print("EXECUTION STATISTICS")
        print("="*60)
        print(f"Execution time: {stats['execution_time']:.2f}s")
        print(f"Number of waypoints: {stats['num_waypoints']}")
        if stats['final_ee_position'] is not None:
            print(f"Final EE position: [{stats['final_ee_position'][0]:.3f}, "
                  f"{stats['final_ee_position'][1]:.3f}, "
                  f"{stats['final_ee_position'][2]:.3f}]")
        if stats['target_error'] is not None:
            print(f"Target error: {stats['target_error']*1000:.1f}mm")
            print(f"Success: {'Yes' if stats['success'] else 'No'}")
        print("="*60)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Load and visualize saved trajectories from Pareto search')
    
    parser.add_argument('experiment_dir', type=str,
                        help='Path to the experiment directory containing saved trajectories')
    parser.add_argument('--trajectory-id', type=str, default=None,
                        help='Specific trajectory ID to visualize (if not provided, will list available trajectories)')
    parser.add_argument('--list', action='store_true',
                        help='List available trajectories and exit')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Check if experiment directory exists
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        return
        
    try:
        # Load experiment data
        loader = TrajectoryLoader(args.experiment_dir)
        
        # If just listing trajectories, print and exit
        if args.list:
            loader.list_trajectories()
            return
            
        # If no specific trajectory ID provided, list available trajectories
        if args.trajectory_id is None:
            loader.list_trajectories()
            print("\nTo visualize a specific trajectory, use:")
            print(f"python trajectory_loader.py {args.experiment_dir} --trajectory-id <trajectory_id>")
            return
            
        # Load and visualize the specific trajectory
        trajectory_data = loader.load_trajectory(args.trajectory_id)
        if trajectory_data is None:
            print(f"Available trajectories:")
            loader.list_trajectories()
            return
            
        # Create visualization demo and run
        demo = TrajectoryVisualizationDemo(loader)
        demo.visualize_trajectory(trajectory_data)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()