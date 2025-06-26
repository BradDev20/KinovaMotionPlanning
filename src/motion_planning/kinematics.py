"""
Kinematics module using MuJoCo for forward/inverse kinematics
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
from typing import Tuple, Optional


class KinematicsSolver:
    """Kinematics solver using MuJoCo for the Kinova Gen3 robot"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Get end-effector body ID (bracelet_link for Kinova Gen3)
        self.end_effector_id = 8  # bracelet_link from model inspection
        
        # Proper joint limits for Kinova Gen3 (in radians)
        # Based on actual robot specs: some joints unlimited, others have physical limits
        self.joint_limits_lower = np.array([
            -np.inf,  # joint_1: unlimited rotation
            -2.24,    # joint_2: shoulder pitch
            -np.inf,  # joint_3: unlimited rotation  
            -2.57,    # joint_4: elbow flex
            -np.inf,  # joint_5: unlimited rotation
            -2.09,    # joint_6: wrist pitch
            -np.inf   # joint_7: unlimited rotation
        ])
        
        self.joint_limits_upper = np.array([
            np.inf,   # joint_1: unlimited rotation
            2.24,     # joint_2: shoulder pitch
            np.inf,   # joint_3: unlimited rotation
            2.57,     # joint_4: elbow flex
            np.inf,   # joint_5: unlimited rotation
            2.09,     # joint_6: wrist pitch
            np.inf    # joint_7: unlimited rotation
        ])
        
        # For optimization, use reasonable bounds for unlimited joints
        self.optimization_limits_lower = np.array([-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14])
        self.optimization_limits_upper = np.array([3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14])
        
        # Backup current state
        self._backup_qpos = None
        self._backup_qvel = None
    
    def _backup_state(self):
        """Backup current MuJoCo state"""
        self._backup_qpos = self.data.qpos.copy()
        self._backup_qvel = self.data.qvel.copy()
    
    def _restore_state(self):
        """Restore MuJoCo state"""
        if self._backup_qpos is not None:
            self.data.qpos[:] = self._backup_qpos
            self.data.qvel[:] = self._backup_qvel
            mujoco.mj_forward(self.model, self.data)
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint angles
        
        Args:
            joint_angles: 7-DOF joint angles in radians
            
        Returns:
            position: 3D position of end-effector
            orientation: 3x3 rotation matrix of end-effector
        """
        # Backup current state
        self._backup_state()
        
        # Set joint angles
        self.data.qpos[:7] = joint_angles
        
        # Compute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get end-effector pose
        position = self.data.xpos[self.end_effector_id].copy()
        orientation = self.data.xmat[self.end_effector_id].reshape(3, 3).copy()
        
        # Restore state
        self._restore_state()
        
        return position, orientation
    
    def inverse_kinematics(self, 
                          target_position: np.ndarray, 
                          target_orientation: Optional[np.ndarray] = None,
                          initial_guess: Optional[np.ndarray] = None,
                          position_tolerance: float = 1e-4,
                          orientation_tolerance: float = 1e-3) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for a target SE(3) pose
        
        Args:
            target_position: Target 3D position
            target_orientation: Target 3x3 rotation matrix (optional)
            initial_guess: Initial joint configuration (default: current state)
            position_tolerance: Position error tolerance
            orientation_tolerance: Orientation error tolerance
            
        Returns:
            joint_angles: Solution joint angles (7-DOF)
            success: Whether IK converged successfully
        """
        if initial_guess is None:
            initial_guess = self.data.qpos[:7].copy()
        
        def objective_function(joint_angles):
            """Objective function for IK optimization"""
            pos, rot = self.forward_kinematics(joint_angles)
            
            # Position error
            pos_error = np.linalg.norm(pos - target_position)
            
            # Orientation error (if target orientation provided)
            rot_error = 0.0
            if target_orientation is not None:
                # Use Frobenius norm of rotation difference
                rot_diff = rot - target_orientation
                rot_error = np.linalg.norm(rot_diff)
            
            return pos_error + rot_error
        
        # Joint limit constraints (use optimization bounds to avoid infinite bounds)
        bounds = [(low, high) for low, high in zip(self.optimization_limits_lower, self.optimization_limits_upper)]
        
        # Solve optimization
        result = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 500}  # Relaxed tolerance, fewer iterations
        )
        
        # Check success
        final_pos, final_rot = self.forward_kinematics(result.x)
        pos_error = np.linalg.norm(final_pos - target_position)
        
        rot_error = 0.0
        if target_orientation is not None:
            rot_error = np.linalg.norm(final_rot - target_orientation)
        
        success = (pos_error < position_tolerance and 
                  rot_error < orientation_tolerance)
        
        return result.x, bool(success)  # Ensure bool return type
    
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose"""
        position = self.data.xpos[self.end_effector_id].copy()
        orientation = self.data.xmat[self.end_effector_id].reshape(3, 3).copy()
        return position, orientation
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits for planning"""
        return self.joint_limits_lower.copy(), self.joint_limits_upper.copy() 