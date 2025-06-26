"""
Integration module for motion planning with MuJoCo
"""

import numpy as np
import mujoco
from typing import List, Tuple, Optional, Callable
from .kinematics import KinematicsSolver
from .planners import RRTPlanner, MotionPlannerFactory


class MotionPlanningInterface:
    """High-level interface for SE(3) goal planning with MuJoCo"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize motion planning interface
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Initialize components
        self.kinematics = KinematicsSolver(model, data)
        self.planner = MotionPlannerFactory.create_rrt_planner(model, data)
        
        # Setup collision checking
        collision_checker = MotionPlannerFactory.create_collision_checker(model, data)
        self.planner.set_collision_checker(collision_checker)
        
        # Trajectory execution parameters
        self.execution_speed = 1.0  # Speed multiplier for trajectory execution
        
    def plan_to_pose(self, 
                    target_position: np.ndarray,
                    target_orientation: Optional[np.ndarray] = None,
                    initial_guess: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], bool]:
        """
        Plan a trajectory to reach a target SE(3) pose
        
        Args:
            target_position: Target 3D position [x, y, z]
            target_orientation: Target 3x3 rotation matrix (optional)
            initial_guess: Initial guess for IK (optional)
            
        Returns:
            trajectory: List of joint configurations forming the trajectory
            success: Whether planning succeeded
        """
        print(f"Planning to target position: {target_position}")
        if target_orientation is not None:
            print(f"With target orientation specified")
        
        # Step 1: Solve inverse kinematics for goal pose
        goal_joints, ik_success = self.kinematics.inverse_kinematics(
            target_position, 
            target_orientation, 
            initial_guess
        )
        
        if not ik_success:
            print("❌ Inverse kinematics failed")
            return [], False
        
        print(f"✅ IK succeeded: goal joints = {[f'{x:.3f}' for x in goal_joints]}")
        
        # Step 2: Plan path from current configuration to goal
        current_joints = self.data.qpos[:7].copy()
        
        print(f"Planning path from current joints: {[f'{x:.3f}' for x in current_joints]}")
        
        trajectory, planning_success = self.planner.plan(current_joints, goal_joints)
        
        if not planning_success:
            print("❌ Path planning failed")
            return [], False
        
        print(f"✅ Path planning succeeded: {len(trajectory)} waypoints")
        
        # Step 3: Smooth trajectory
        smoothed_trajectory = self.planner.smooth_path(trajectory)
        print(f"✅ Path smoothed: {len(smoothed_trajectory)} waypoints")
        
        return smoothed_trajectory, True
    
    def plan_to_joint_configuration(self, target_joints: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Plan a trajectory to a target joint configuration
        
        Args:
            target_joints: Target 7-DOF joint configuration
            
        Returns:
            trajectory: List of joint configurations forming the trajectory
            success: Whether planning succeeded
        """
        print(f"Planning to target joints: {[f'{x:.3f}' for x in target_joints]}")
        
        current_joints = self.data.qpos[:7].copy()
        trajectory, success = self.planner.plan(current_joints, target_joints)
        
        if success:
            smoothed_trajectory = self.planner.smooth_path(trajectory)
            print(f"✅ Joint space planning succeeded: {len(smoothed_trajectory)} waypoints")
            return smoothed_trajectory, True
        else:
            print("❌ Joint space planning failed")
            return [], False
    
    def execute_trajectory(self, 
                          trajectory: List[np.ndarray], 
                          callback: Optional[Callable] = None,
                          dt: float = 0.01) -> bool:
        """
        Execute a trajectory by updating MuJoCo control signals
        
        Args:
            trajectory: List of joint configurations to execute
            callback: Optional callback function called at each step
            dt: Time step for execution
            
        Returns:
            success: Whether execution completed successfully
        """
        if not trajectory:
            print("❌ Empty trajectory provided")
            return False
        
        print(f"Executing trajectory with {len(trajectory)} waypoints...")
        
        try:
            for i, waypoint in enumerate(trajectory):
                # Set control signals to follow waypoint
                self.data.ctrl[:7] = waypoint
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Call callback if provided
                if callback:
                    callback(i, waypoint, self.data)
                    
                # Small delay for visualization
                if dt > 0:
                    import time
                    time.sleep(dt * self.execution_speed)
            
            print("✅ Trajectory execution completed")
            return True
            
        except Exception as e:
            print(f"❌ Trajectory execution failed: {e}")
            return False
    
    def get_current_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose"""
        return self.kinematics.get_current_pose()
    
    def validate_trajectory(self, trajectory: List[np.ndarray]) -> bool:
        """
        Validate that trajectory is collision-free and executable
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            valid: Whether trajectory is valid
        """
        if not trajectory:
            return False
        
        collision_checker = self.planner.collision_checker
        if collision_checker is None:
            return True  # No collision checking available
        
        for waypoint in trajectory:
            if collision_checker(waypoint):
                return False
        
        return True
    
    def set_planning_parameters(self, 
                              step_size: Optional[float] = None,
                              max_iterations: Optional[int] = None,
                              goal_threshold: Optional[float] = None):
        """Update planning parameters"""
        if step_size is not None:
            self.planner.step_size = step_size
        if max_iterations is not None:
            self.planner.max_iterations = max_iterations
        if goal_threshold is not None:
            self.planner.goal_threshold = goal_threshold
    
    def set_execution_speed(self, speed: float):
        """Set execution speed multiplier"""
        self.execution_speed = max(0.1, min(10.0, speed))  # Clamp between 0.1x and 10x


class TrajectoryVisualizer:
    """Utility class for visualizing trajectories in MuJoCo"""
    
    @staticmethod
    def visualize_trajectory(model: mujoco.MjModel, 
                           data: mujoco.MjData,
                           trajectory: List[np.ndarray],
                           kinematics: KinematicsSolver,
                           num_samples: int = 10) -> List[np.ndarray]:
        """
        Visualize trajectory by computing end-effector positions
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            trajectory: Joint space trajectory
            kinematics: Kinematics solver
            num_samples: Number of samples to visualize
            
        Returns:
            end_effector_positions: List of 3D positions along trajectory
        """
        if not trajectory:
            return []
        
        # Sample trajectory points
        step = max(1, len(trajectory) // num_samples)
        sampled_trajectory = trajectory[::step]
        
        positions = []
        for joints in sampled_trajectory:
            pos, _ = kinematics.forward_kinematics(joints)
            positions.append(pos)
        
        return positions 