"""
Integration module for motion planning with MuJoCo
"""

import numpy as np
import mujoco
from typing import List, Tuple, Optional, Callable
from .kinematics import KinematicsSolver
from .planners import RRTPlanner, MotionPlannerFactory


class MotionPlanningInterface:
    """High-level interface for motion planning with the Kinova Gen3"""
    
    def __init__(self, model_path: str):
        """
        Initialize motion planning interface
        
        Args:
            model_path: Path to MuJoCo XML model file
        """
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize components with new API
        self.kinematics = KinematicsSolver(model_path)
        self.planner = RRTPlanner(self.model, self.data)
        
        # Setup collision checking
        collision_checker = MotionPlannerFactory.create_collision_checker(self.model, self.data)
        self.planner.set_collision_checker(collision_checker)
        
        # Trajectory execution parameters
        self.execution_speed = 1.0  # Speed multiplier for trajectory execution
        
    def plan_to_pose(self, 
                     target_position: np.ndarray,
                     target_orientation: Optional[np.ndarray] = None,
                     start_config: Optional[np.ndarray] = None,
                     **kwargs) -> Tuple[Optional[List[np.ndarray]], bool]:
        """
        Plan a trajectory to reach a target SE(3) pose
        
        Args:
            target_position: Target 3D position for gripper center
            target_orientation: Target 3x3 rotation matrix (optional)
            start_config: Starting joint configuration (default: current)
            **kwargs: Additional arguments for planning
            
        Returns:
            trajectory: List of joint configurations, or None if planning failed
            success: Whether planning succeeded
        """
        if start_config is None:
            start_config = self.data.qpos[:7].copy()
            
        # Solve inverse kinematics for target pose
        target_config, ik_success = self.kinematics.inverse_kinematics(
            target_position, target_orientation, initial_guess=start_config
        )
        
        if not ik_success:
            print(f"IK failed for target pose: {target_position}")
            return None, False
            
        # Plan path in joint space
        path, planning_success = self.planner.plan(start_config, target_config)
        
        return path, planning_success
        
    def plan_to_joint_config(self,
                           target_config: np.ndarray,
                           start_config: Optional[np.ndarray] = None) -> Tuple[Optional[List[np.ndarray]], bool]:
        """
        Plan a trajectory to reach a target joint configuration
        
        Args:
            target_config: Target joint configuration (7-DOF)
            start_config: Starting joint configuration (default: current)
            
        Returns:
            trajectory: List of joint configurations, or None if planning failed  
            success: Whether planning succeeded
        """
        if start_config is None:
            start_config = self.data.qpos[:7].copy()
            
        return self.planner.plan(start_config, target_config)
        
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current gripper center pose"""
        return self.kinematics.get_current_pose()
        
    def execute_trajectory(self, trajectory: List[np.ndarray], dt: float = 0.01):
        """
        Execute a trajectory in simulation
        
        Args:
            trajectory: List of joint configurations
            dt: Time step between waypoints
        """
        for config in trajectory:
            # Set joint positions 
            self.data.qpos[:7] = config
            self.data.ctrl[:7] = config  # PD control to maintain position
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Optional: add delay for visualization
            import time
            time.sleep(dt)
    
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