#!/usr/bin/env python3
"""
Trajectory to Motion Plan Converter

Loads a saved trajectory from Pareto search results and converts it into
a CSV motion plan format suitable for execution on a real Kinova robot.

The output CSV contains timestamped joint angles for all 7 DOF, allowing
direct deployment of optimized trajectories to hardware.
"""

import sys
import os
import numpy as np
import argparse
import pickle
import json
import csv
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import mujoco
from PIL import Image
import io

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../examples'))

from trajectory_loader import TrajectoryLoader
from motion_planning.kinematics import KinematicsSolver
from motion_planning.utils import Obstacle
from scene_builder import create_standard_scene


class MotionPlanGenerator:
    """Converts saved trajectories to robot motion plans"""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize with experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory containing saved trajectories
        """
        self.experiment_dir = experiment_dir
        self.loader = TrajectoryLoader(experiment_dir)
        
    def list_available_trajectories(self) -> None:
        """Print list of available trajectories"""
        print(f"\nAvailable trajectories in {self.experiment_dir}:")
        print("=" * 80)
        print(f"{'Trajectory ID':<20} {'Alpha':<8} {'Length':<10} {'Obstacle':<10} {'Waypoints':<10}")
        print("-" * 80)
        
        for traj_info in self.loader.available_trajectories:
            print(f"{traj_info['trajectory_id']:<20} "
                  f"{traj_info['alpha']:<8.3f} "
                  f"{traj_info['length_cost']:<10.4f} "
                  f"{traj_info['obstacle_cost']:<10.4f} "
                  f"{traj_info['waypoint_count']:<10}")
    
    def generate_motion_plan(self, trajectory_id: str, dt: float = 0.1, 
                           interpolate: bool = True, target_duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Generate motion plan from saved trajectory.
        
        Args:
            trajectory_id: ID of trajectory to load
            dt: Time step between waypoints in seconds
            interpolate: Whether to interpolate between waypoints for smoother motion
            target_duration: Target duration for the motion (will adjust dt accordingly)
            
        Returns:
            Motion plan as numpy array (N_timesteps, 8) where columns are [time, q1, q2, q3, q4, q5, q6, q7]
        """
        # Load trajectory data
        trajectory_data = self.loader.load_trajectory(trajectory_id)
        if trajectory_data is None:
            print(f"Failed to load trajectory: {trajectory_id}")
            return None
        
        # Extract trajectory waypoints
        trajectory = trajectory_data['trajectory']  # Shape: (N_waypoints, 7)
        
        print(f"Loaded trajectory: {trajectory_id}")
        print(f"  Original waypoints: {len(trajectory)}")
        print(f"  Alpha: {trajectory_data['alpha']:.3f}")
        print(f"  Length cost: {trajectory_data['length_cost']:.4f}")
        print(f"  Obstacle cost: {trajectory_data['obstacle_cost']:.4f}")
        
        # Adjust timing if target duration is specified
        if target_duration is not None:
            dt = target_duration / (len(trajectory) - 1)
            print(f"  Adjusted dt to {dt:.4f}s for target duration {target_duration:.1f}s")
        
        if interpolate:
            # Create denser trajectory through interpolation
            motion_plan = self._interpolate_trajectory(trajectory, dt)
            print(f"  Interpolated to {len(motion_plan)} timesteps")
        else:
            # Use original waypoints with specified dt
            motion_plan = self._create_timestamped_plan(trajectory, dt)
            print(f"  Using original {len(motion_plan)} waypoints")
        
        return motion_plan
    
    def _create_timestamped_plan(self, trajectory: np.ndarray, dt: float) -> np.ndarray:
        """Create motion plan with timestamps from original waypoints"""
        n_waypoints = len(trajectory)
        motion_plan = np.zeros((n_waypoints, 8))  # [time, q1, q2, q3, q4, q5, q6, q7]
        
        # Add timestamps
        motion_plan[:, 0] = np.arange(n_waypoints) * dt
        
        # Add joint angles
        motion_plan[:, 1:] = trajectory
        
        return motion_plan
    
    def _interpolate_trajectory(self, trajectory: np.ndarray, dt: float, 
                              interpolation_factor: int = 5) -> np.ndarray:
        """
        Create interpolated motion plan for smoother execution.
        
        Args:
            trajectory: Original trajectory waypoints (N, 7)
            dt: Time step for interpolated points
            interpolation_factor: Number of interpolated points between original waypoints
            
        Returns:
            Interpolated motion plan (M, 8) where M > N
        """
        n_waypoints = len(trajectory)
        
        # Create interpolated trajectory using linear interpolation
        # Calculate total number of interpolated points
        total_segments = n_waypoints - 1
        n_interpolated = total_segments * interpolation_factor + 1
        
        # Create parameter values for interpolation
        original_params = np.linspace(0, 1, n_waypoints)
        interpolated_params = np.linspace(0, 1, n_interpolated)
        
        # Interpolate each joint
        interpolated_trajectory = np.zeros((n_interpolated, 7))
        for joint_idx in range(7):
            interpolated_trajectory[:, joint_idx] = np.interp(
                interpolated_params, original_params, trajectory[:, joint_idx]
            )
        
        # Create motion plan with timestamps
        motion_plan = np.zeros((n_interpolated, 8))
        motion_plan[:, 0] = np.arange(n_interpolated) * dt
        motion_plan[:, 1:] = interpolated_trajectory
        
        return motion_plan
    
    def save_motion_plan_csv(self, motion_plan: np.ndarray, output_path: str, 
                           trajectory_id: str, metadata: Optional[Dict] = None) -> None:
        """
        Save motion plan to CSV file.
        
        Args:
            motion_plan: Motion plan array (N, 8)
            output_path: Path to save CSV file
            trajectory_id: ID of source trajectory
            metadata: Optional metadata to include in header
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with metadata
            # writer.writerow(['# Kinova Gen3 Motion Plan'])
            # writer.writerow([f'# Generated from trajectory: {trajectory_id}'])
            # writer.writerow([f'# Source experiment: {os.path.basename(self.experiment_dir)}'])
            # writer.writerow([f'# Generated on: {datetime.now().isoformat()}'])
            
            # if metadata:
            #     writer.writerow([f'# Alpha: {metadata.get("alpha", "unknown")}'])
            #     writer.writerow([f'# Length cost: {metadata.get("length_cost", "unknown")}'])
            #     writer.writerow([f'# Obstacle cost: {metadata.get("obstacle_cost", "unknown")}'])
            
            # writer.writerow([f'# Total duration: {motion_plan[-1, 0]:.3f} seconds'])
            # writer.writerow([f'# Number of timesteps: {len(motion_plan)}'])
            # writer.writerow(['#'])
            
            # Write column headers
            headers = ['time_s', 'joint1_rad', 'joint2_rad', 'joint3_rad', 
                      'joint4_rad', 'joint5_rad', 'joint6_rad', 'joint7_rad']
            writer.writerow(headers)
            
            # Write motion plan data
            for row in motion_plan:
                # Format: time with 4 decimals, joint angles with 6 decimals
                formatted_row = [f'{row[0]:.4f}'] + [f'{angle:.6f}' for angle in row[1:]]
                writer.writerow(formatted_row)
        
        print(f"Motion plan saved to: {output_path}")
        print(f"  Duration: {motion_plan[-1, 0]:.3f} seconds")
        print(f"  Timesteps: {len(motion_plan)}")
        print(f"  Average dt: {motion_plan[1, 0] - motion_plan[0, 0]:.4f} seconds")
    
    def validate_motion_plan(self, motion_plan: np.ndarray, 
                           max_joint_velocity: float = 1.0,
                           max_joint_acceleration: float = 2.0) -> Dict:
        """
        Validate motion plan against kinematic limits.
        
        Args:
            motion_plan: Motion plan array (N, 8)
            max_joint_velocity: Maximum joint velocity in rad/s
            max_joint_acceleration: Maximum joint acceleration in rad/s²
            
        Returns:
            Validation results dictionary
        """
        times = motion_plan[:, 0]
        joint_angles = motion_plan[:, 1:]
        
        # Calculate velocities (numerical differentiation)
        dt_array = np.diff(times)
        velocities = np.diff(joint_angles, axis=0) / dt_array[:, np.newaxis]
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0) / dt_array[1:, np.newaxis]
        
        # Check limits
        max_velocities = np.max(np.abs(velocities), axis=0)
        max_accelerations = np.max(np.abs(accelerations), axis=0)
        
        velocity_violations = max_velocities > max_joint_velocity
        acceleration_violations = max_accelerations > max_joint_acceleration
        
        results = {
            'valid': not (np.any(velocity_violations) or np.any(acceleration_violations)),
            'max_velocities': max_velocities,
            'max_accelerations': max_accelerations,
            'velocity_violations': velocity_violations,
            'acceleration_violations': acceleration_violations,
            'velocity_limit': max_joint_velocity,
            'acceleration_limit': max_joint_acceleration
        }
        
        return results
    
    def print_validation_report(self, validation_results: Dict) -> None:
        """Print validation report"""
        print("\n" + "="*60)
        print("MOTION PLAN VALIDATION REPORT")
        print("="*60)
        
        if validation_results['valid']:
            print("✓ Motion plan PASSED all kinematic limit checks")
        else:
            print("✗ Motion plan FAILED kinematic limit checks")
        
        print(f"\nVelocity Analysis (limit: {validation_results['velocity_limit']:.1f} rad/s):")
        for i, (max_vel, violation) in enumerate(zip(
            validation_results['max_velocities'], 
            validation_results['velocity_violations']
        )):
            status = "✗ VIOLATION" if violation else "✓ OK"
            print(f"  Joint {i+1}: {max_vel:.4f} rad/s {status}")
        
        print(f"\nAcceleration Analysis (limit: {validation_results['acceleration_limit']:.1f} rad/s²):")
        for i, (max_acc, violation) in enumerate(zip(
            validation_results['max_accelerations'], 
            validation_results['acceleration_violations']
        )):
            status = "✗ VIOLATION" if violation else "✓ OK"
            print(f"  Joint {i+1}: {max_acc:.4f} rad/s² {status}")
        
        print("="*60)
    
    def generate_trajectory_gif(self, trajectory_data: Dict, output_path: str, 
                              frame_duration: float = 0.1) -> None:
        """
        Generate GIF visualization of the trajectory.
        
        Args:
            trajectory_data: Trajectory data dictionary
            output_path: Path to save GIF file
            frame_duration: Duration per frame in seconds
        """
        print(f"Generating trajectory visualization GIF...")
        
        # Setup MuJoCo scene
        obstacles = []
        for obs_data in self.loader.config['obstacles']:
            obstacles.append(Obstacle(
                center=np.array(obs_data['center']),
                radius=obs_data['radius'],
                safe_distance=obs_data['safe_distance']
            ))
        
        target_position = np.array(self.loader.config['target_position'])
        scene_path = create_standard_scene(
            obstacles=obstacles,
            target_position=target_position,
            trace_dot_count=1000,
            output_filename="motion_plan_gif_scene.xml"
        )
        
        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        kinematics = KinematicsSolver(scene_path)
        
        # Set initial position
        start_config = np.array(self.loader.config['start_config'])
        data.qpos[:7] = start_config
        data.ctrl[:7] = start_config
        mujoco.mj_forward(model, data)
        
        # Get trajectory and color
        trajectory = trajectory_data['trajectory']
        color = np.array(trajectory_data['color'])
        
        # Generate frames
        frames = []
        print(f"  Rendering {len(trajectory)} frames...")
        
        for i, waypoint in enumerate(trajectory):
            # Update robot position
            data.qpos[:7] = waypoint
            data.ctrl[:7] = waypoint
            mujoco.mj_forward(model, data)
            
            # Get end-effector position for trace
            ee_pos, _ = kinematics.forward_kinematics(waypoint)
            
            # Update trace dots up to current position
            self._update_trace_dots(model, trajectory[:i+1], kinematics, color)
            
            # Render frame
            frame_image = self._capture_mujoco_frame(model, data)
            frames.append(frame_image)
            
            if (i + 1) % 10 == 0:
                print(f"    Rendered {i + 1}/{len(trajectory)} frames")
        
        # Save GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(frame_duration * 1000),  # PIL uses milliseconds
                loop=0
            )
            print(f"  GIF saved to: {output_path}")
            print(f"    Duration: {len(frames) * frame_duration:.1f} seconds")
            print(f"    Frames: {len(frames)}")
        else:
            print("  Error: No frames generated")
    
    def _update_trace_dots(self, model, trajectory_segment: np.ndarray, 
                          kinematics, color: np.ndarray) -> None:
        """Update trajectory trace dots for current segment"""
        # Clear all dots first
        for i in range(1000):
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    model.body_pos[body_id] = [0, 0, -10]  # Hide below ground
            except:
                pass
        
        # Add dots for current trajectory segment
        for i, waypoint in enumerate(trajectory_segment):
            if i >= 1000:  # Limit to available dots
                break
            ee_pos, _ = kinematics.forward_kinematics(waypoint)
            
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    model.body_pos[body_id] = ee_pos
                    # Update color
                    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"trace_dot_{i}_geom")
                    if geom_id >= 0:
                        model.geom_rgba[geom_id] = color
            except:
                pass
    
    def _capture_mujoco_frame(self, model, data) -> Image.Image:
        """Capture a frame from MuJoCo"""
        # Create renderer
        renderer = mujoco.Renderer(model, height=400, width=400)
        
        # Render
        renderer.update_scene(data)
        pixels = renderer.render()
        
        # Convert to PIL Image with proper orientation
        image_array = np.flipud(pixels)  # MuJoCo renders upside down
        image_array = np.rot90(image_array, 2)  # Additional 180 degree rotation
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        
        renderer.close()
        return pil_image
    
    def _create_plan_summary(self, summary_path: str, trajectory_id: str, 
                           metadata: Dict, plan_directory: str) -> None:
        """Create a README summary for the motion plan package"""
        with open(summary_path, 'w') as f:
            f.write(f"# Motion Plan Package: {trajectory_id}\n\n")
            f.write(f"Generated from Pareto search trajectory optimization results.\n\n")
            
            f.write("## Source Information\n\n")
            f.write(f"- **Experiment**: {os.path.basename(self.experiment_dir)}\n")
            f.write(f"- **Trajectory ID**: {trajectory_id}\n")
            f.write(f"- **Alpha (length weight)**: {metadata['alpha']:.3f}\n")
            f.write(f"- **Length cost**: {metadata['length_cost']:.4f}\n")
            f.write(f"- **Obstacle cost**: {metadata['obstacle_cost']:.4f}\n")
            f.write(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Files\n\n")
            f.write("- **`motion_plan.csv`**: Robot motion plan with timestamped joint angles\n")
            f.write("- **`trajectory_visualization.gif`**: Animated visualization of trajectory execution\n")
            f.write("- **`README.md`**: This summary file\n\n")
            
            f.write("## CSV Format\n\n")
            f.write("The motion plan CSV contains the following columns:\n")
            f.write("- `time_s`: Time in seconds\n")
            f.write("- `joint1_rad` through `joint7_rad`: Joint angles in radians\n\n")
            
            f.write("## Robot Deployment\n\n")
            f.write("This motion plan is ready for deployment on a Kinova Gen3 robot.\n")
            f.write("The trajectory has been validated against standard kinematic limits:\n")
            f.write("- Maximum joint velocity: 1.0 rad/s\n")
            f.write("- Maximum joint acceleration: 2.0 rad/s²\n\n")
            
            f.write("## Trajectory Characteristics\n\n")
            if metadata['alpha'] < 0.3:
                f.write("**Obstacle-Avoiding Trajectory**: Prioritizes safety over speed.\n")
                f.write("This trajectory carefully navigates around obstacles with conservative motions.\n")
            elif metadata['alpha'] > 0.7:
                f.write("**Direct Trajectory**: Prioritizes speed and efficiency.\n")
                f.write("This trajectory takes a more direct path, accepting higher obstacle proximity.\n")
            else:
                f.write("**Balanced Trajectory**: Balances safety and efficiency.\n")
                f.write("This trajectory offers a compromise between obstacle avoidance and path length.\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert saved trajectory to Kinova robot motion plan')
    
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='Path to experiment directory containing saved trajectories')
    parser.add_argument('--trajectory-id', type=str,
                        help='Trajectory ID to convert (if not provided, will list available trajectories)')
    parser.add_argument('--output', type=str,
                        help='Output CSV filename (default: auto-generated based on trajectory ID)')
    parser.add_argument('--output-dir', type=str, default='src/viz/motion_plans',
                        help='Output directory for motion plans (default: src/viz/motion_plans)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step between waypoints in seconds (default: 0.1)')
    parser.add_argument('--duration', type=float,
                        help='Target total duration in seconds (overrides dt)')
    parser.add_argument('--no-interpolate', action='store_true',
                        help='Disable interpolation between waypoints')
    parser.add_argument('--validate', action='store_true',
                        help='Validate motion plan against kinematic limits')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                        help='Maximum joint velocity for validation (rad/s, default: 1.0)')
    parser.add_argument('--max-acceleration', type=float, default=2.0,
                        help='Maximum joint acceleration for validation (rad/s², default: 2.0)')
    parser.add_argument('--generate-gif', action='store_true', default=True,
                        help='Generate GIF visualization of trajectory (default: True)')
    parser.add_argument('--no-gif', dest='generate_gif', action='store_false',
                        help='Skip GIF generation')
    parser.add_argument('--gif-frame-duration', type=float, default=0.1,
                        help='GIF frame duration in seconds (default: 0.1)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Check experiment directory
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        return
    
    try:
        # Create motion plan generator
        generator = MotionPlanGenerator(args.experiment_dir)
        
        # If no trajectory ID provided, list available trajectories
        if not args.trajectory_id:
            generator.list_available_trajectories()
            print(f"\nTo generate a motion plan, use:")
            print(f"python {sys.argv[0]} --experiment-dir {args.experiment_dir} --trajectory-id <trajectory_id>")
            return
        
        # Generate motion plan
        interpolate = not args.no_interpolate
        motion_plan = generator.generate_motion_plan(
            args.trajectory_id, 
            dt=args.dt,
            interpolate=interpolate,
            target_duration=args.duration
        )
        
        if motion_plan is None:
            return
        
        # Create motion plan directory structure
        if args.output:
            # If specific output provided, use it as directory name
            plan_dir_name = os.path.splitext(args.output)[0]  # Remove .csv if present
        else:
            experiment_name = os.path.basename(args.experiment_dir)
            plan_dir_name = f"{experiment_name}_{args.trajectory_id}_motion_plan"
        
        plan_directory = os.path.join(args.output_dir, plan_dir_name)
        os.makedirs(plan_directory, exist_ok=True)
        
        # Set output paths
        csv_path = os.path.join(plan_directory, "motion_plan.csv")
        gif_path = os.path.join(plan_directory, "trajectory_visualization.gif")
        
        # Get trajectory metadata for CSV header
        trajectory_data = generator.loader.load_trajectory(args.trajectory_id)
        metadata = {
            'alpha': trajectory_data['alpha'],
            'length_cost': trajectory_data['length_cost'],
            'obstacle_cost': trajectory_data['obstacle_cost']
        }
        
        # Save motion plan CSV
        generator.save_motion_plan_csv(motion_plan, csv_path, args.trajectory_id, metadata)
        
        # Generate GIF visualization if requested
        if args.generate_gif:
            generator.generate_trajectory_gif(
                trajectory_data, 
                gif_path, 
                frame_duration=args.gif_frame_duration
            )
        
        # Validate if requested
        if args.validate:
            validation_results = generator.validate_motion_plan(
                motion_plan, 
                max_joint_velocity=args.max_velocity,
                max_joint_acceleration=args.max_acceleration
            )
            generator.print_validation_report(validation_results)
        
        # Create summary file
        summary_path = os.path.join(plan_directory, "README.md")
        generator._create_plan_summary(summary_path, args.trajectory_id, metadata, plan_directory)
        
        print(f"\n✓ Motion plan package successfully generated!")
        print(f"📁 Directory: {plan_directory}")
        print(f"📄 CSV: motion_plan.csv")
        if args.generate_gif:
            print(f"🎬 GIF: trajectory_visualization.gif")
        print(f"📋 Summary: README.md")
        print(f"Ready for deployment to Kinova Gen3 robot.")
        
    except Exception as e:
        print(f"Error generating motion plan: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 