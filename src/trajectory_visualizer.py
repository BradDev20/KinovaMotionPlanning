#!/usr/bin/env python3
"""
Trajectory Visualization Manager for MuJoCo Demonstrations

This module provides a reusable manager class for executing and visualizing
robot trajectories in MuJoCo with:
- Smooth trajectory execution
- Real-time trajectory tracing with colored dots
- Interactive viewer controls (replay, clear, home, quit)
- Real-time analysis and feedback

Used by trajectory optimization demos to avoid code duplication.
"""

import mujoco
import numpy as np
import time
from typing import List, Optional, Callable, Any


class TrajectoryVisualizationManager:
    """Manager class for executing and visualizing trajectories in MuJoCo viewer"""
    
    def __init__(self, model, data, viewer_handle, kinematics_solver):
        """
        Initialize the trajectory visualization manager.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            viewer_handle: MuJoCo viewer handle
            kinematics_solver: Kinematics solver for forward kinematics
        """
        self.model = model
        self.data = data
        self.viewer_handle = viewer_handle
        self.kinematics = kinematics_solver
        self.current_trajectory = None
        self.target_position = None
        
    def set_target_position(self, target_position: np.ndarray):
        """Set the target position for analysis"""
        self.target_position = target_position
        
    def clear_trajectory_trace(self, max_dots: int = 500):
        """Hide all trajectory trace dots by moving them below ground"""
        for i in range(max_dots):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    self.model.body_pos[body_id] = [0, 0, -10]  # Move far below ground
            except:
                pass  # Body doesn't exist
                
    def update_trajectory_trace(self, ee_positions: List[np.ndarray], 
                               color: Optional[np.ndarray] = None,
                               start_dot_offset: int = 0):
        """
        Update trajectory trace dots to show end-effector path.
        
        Args:
            ee_positions: List of end-effector positions
            color: RGBA color array for the trace dots (optional)
            start_dot_offset: Starting dot index offset for multiple trajectories
        """
        for i, pos in enumerate(ee_positions[-200:]):  # Show last 200 points
            dot_id = start_dot_offset + i
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{dot_id}")
                if body_id >= 0:
                    self.model.body_pos[body_id] = pos
                    
                    # Update color if provided
                    if color is not None:
                        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"trace_dot_{dot_id}_geom")
                        if geom_id >= 0:
                            self.model.geom_rgba[geom_id] = color
            except:
                pass  # Skip if dot doesn't exist
                
    def execute_trajectory(self, trajectory: List[np.ndarray], 
                          steps_per_waypoint: int = 8,
                          step_delay: float = 0.002,
                          show_trace: bool = True,
                          initial_pause: float = 1.0,
                          trace_color: Optional[np.ndarray] = None,
                          verbose: bool = True) -> dict:
        """
        Execute a trajectory with smooth animation and optional tracing.
        
        Args:
            trajectory: List of joint configurations
            steps_per_waypoint: Number of simulation steps per waypoint
            step_delay: Delay between simulation steps
            show_trace: Whether to show trajectory trace
            initial_pause: Pause before starting execution
            trace_color: Color for trace dots
            verbose: Whether to print execution feedback
            
        Returns:
            Dictionary with execution statistics
        """
        if show_trace:
            self.clear_trajectory_trace()
            
        # Start at initial position
        self.data.qpos[:7] = trajectory[0]
        self.data.ctrl[:7] = trajectory[0]
        mujoco.mj_forward(self.model, self.data)
        self.viewer_handle.sync()
        time.sleep(initial_pause)
        
        if verbose:
            print(f"Executing trajectory with {len(trajectory)} waypoints...")
        
        # Track end-effector positions for tracing
        ee_positions = []
        execution_start = time.time()
        
        for i, waypoint in enumerate(trajectory):
            # Set robot configuration
            self.data.qpos[:7] = waypoint
            self.data.ctrl[:7] = waypoint
            
            # Get current end-effector position
            current_ee_pos, _ = self.kinematics.forward_kinematics(waypoint)
            ee_positions.append(current_ee_pos.copy())
            
            # Update trajectory trace periodically
            if show_trace and (i % max(1, len(trajectory)//50) == 0 or i == len(trajectory)-1):
                self.update_trajectory_trace(ee_positions, trace_color)
            
            # Step simulation for smooth visualization
            for _ in range(steps_per_waypoint):
                mujoco.mj_step(self.model, self.data)
                self.viewer_handle.sync()
                time.sleep(step_delay)
        
        # Final trace update
        if show_trace:
            self.update_trajectory_trace(ee_positions, trace_color)
        
        execution_time = time.time() - execution_start
        
        # Calculate final statistics
        stats = self._analyze_trajectory_execution(trajectory, ee_positions, execution_time)
        
        if verbose:
            self._print_execution_results(stats)
            
        self.current_trajectory = trajectory
        return stats
        
    def _analyze_trajectory_execution(self, trajectory: List[np.ndarray], 
                                    ee_positions: List[np.ndarray],
                                    execution_time: float) -> dict:
        """Analyze trajectory execution and return statistics"""
        stats = {
            'execution_time': execution_time,
            'num_waypoints': len(trajectory),
            'final_ee_position': ee_positions[-1] if ee_positions else None,
            'target_error': None,
            'success': False
        }
        
        if self.target_position is not None and ee_positions:
            final_error = np.linalg.norm(ee_positions[-1] - self.target_position)
            stats['target_error'] = final_error
            stats['success'] = final_error < 0.02  # 2cm tolerance
            
        return stats
        
    def _print_execution_results(self, stats: dict):
        """Print execution results to console"""
        if stats['target_error'] is not None:
            print(f"Target reached! Final error: {stats['target_error']*1000:.1f}mm")
            if stats['success']:
                print("SUCCESS: Target reached within tolerance!")
            else:
                print("Target positioning could be improved")
        else:
            print("Trajectory execution complete!")
            
    def move_to_home_position(self, home_config: np.ndarray):
        """Move robot to home position"""
        self.data.qpos[:7] = home_config
        self.data.ctrl[:7] = home_config
        mujoco.mj_forward(self.model, self.data)
        self.viewer_handle.sync()
        
    def interactive_controls(self, 
                           home_config: np.ndarray,
                           custom_commands: Optional[dict] = None,
                           show_menu: bool = True) -> bool:
        """
        Run interactive control loop.
        
        Args:
            home_config: Home position joint configuration
            custom_commands: Dictionary of additional commands {key: (description, callback)}
            show_menu: Whether to show the control menu
            
        Returns:
            True if user chose to quit, False if continuing
        """
        if show_menu:
            print("=" * 60)
            print("INTERACTIVE CONTROLS:")
            print("  [r] - Replay trajectory")
            print("  [c] - Clear trajectory trace")
            print("  [h] - Show home position")
            if custom_commands:
                for key, (description, _) in custom_commands.items():
                    print(f"  [{key}] - {description}")
            print("  [q] - Quit demo")
            print("=" * 60)
        
        try:
            choice = input("Enter your choice: ").strip().lower()
            
            if choice == 'r' and self.current_trajectory is not None:
                print("Replaying trajectory...")
                self.execute_trajectory(self.current_trajectory)
                
            elif choice == 'c':
                print("Clearing trajectory trace...")
                self.clear_trajectory_trace()
                mujoco.mj_forward(self.model, self.data)
                self.viewer_handle.sync()
                
            elif choice == 'h':
                print("Moving to home position...")
                self.move_to_home_position(home_config)
                
            elif choice == 'q':
                print("Exiting demo...")
                return True
                
            elif custom_commands and choice in custom_commands:
                _, callback = custom_commands[choice]
                callback()
                
            else:
                print("Invalid choice. Please try again.")
                
            # Keep simulation running
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
                self.viewer_handle.sync()
                time.sleep(0.01)
                
            return False
                
        except (KeyboardInterrupt, EOFError):
            print("Exiting demo...")
            return True
            
    def run_interactive_session(self, 
                               trajectory: List[np.ndarray],
                               home_config: np.ndarray,
                               target_position: Optional[np.ndarray] = None,
                               custom_commands: Optional[dict] = None):
        """
        Run complete interactive session with trajectory execution and controls.
        
        Args:
            trajectory: Trajectory to execute and replay
            home_config: Home position joint configuration  
            target_position: Target position for analysis
            custom_commands: Additional interactive commands
        """
        if target_position is not None:
            self.set_target_position(target_position)
            
        # Execute trajectory initially
        print("Executing trajectory...")
        self.execute_trajectory(trajectory)
        
        # Run interactive control loop
        print("Trajectory execution complete!")
        while True:
            should_quit = self.interactive_controls(home_config, custom_commands)
            if should_quit:
                break


class MultiTrajectoryVisualizer(TrajectoryVisualizationManager):
    """Extended visualizer for handling multiple trajectories with different colors"""
    
    def __init__(self, model, data, viewer_handle, kinematics_solver):
        super().__init__(model, data, viewer_handle, kinematics_solver)
        self.trajectories = []
        self.trajectory_colors = []
        
    def add_trajectory(self, trajectory: List[np.ndarray], color: np.ndarray):
        """Add a trajectory with its color for visualization"""
        self.trajectories.append(trajectory)
        self.trajectory_colors.append(color)
        
    def visualize_all_trajectories(self, verbose: bool = True):
        """Visualize all added trajectories simultaneously"""
        if verbose:
            print(f"Rendering {len(self.trajectories)} trajectories...")
            
        dot_offset = 0
        for i, (trajectory, color) in enumerate(zip(self.trajectories, self.trajectory_colors)):
            # Convert trajectory to end-effector positions
            ee_positions = []
            for waypoint in trajectory:
                ee_pos, _ = self.kinematics.forward_kinematics(waypoint)
                ee_positions.append(ee_pos)
            
            # Render trajectory trace with appropriate color
            self.update_trajectory_trace(ee_positions, color, dot_offset)
            dot_offset += len(ee_positions)
        
        # Update viewer
        mujoco.mj_forward(self.model, self.data)
        self.viewer_handle.sync()
        
        if verbose:
            print("All trajectories rendered!")


def create_trajectory_visualizer(model, data, viewer_handle, kinematics_solver, 
                                multi_trajectory: bool = False):
    """
    Factory function to create appropriate trajectory visualizer.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data  
        viewer_handle: MuJoCo viewer handle
        kinematics_solver: Kinematics solver
        multi_trajectory: Whether to create multi-trajectory visualizer
        
    Returns:
        TrajectoryVisualizationManager or MultiTrajectoryVisualizer instance
    """
    if multi_trajectory:
        return MultiTrajectoryVisualizer(model, data, viewer_handle, kinematics_solver)
    else:
        return TrajectoryVisualizationManager(model, data, viewer_handle, kinematics_solver) 