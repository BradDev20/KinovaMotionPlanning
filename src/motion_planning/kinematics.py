"""
Kinematics module using MuJoCo for forward/inverse kinematics
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
from typing import Tuple, Optional


class KinematicsSolver:
    """Kinematics solver using MuJoCo for the Kinova Gen3 robot"""
    
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # Use gripper center instead of just bracelet_link
        self.end_effector_body = 10  # gripper_base for reference
        self.right_pad_body = 16     # right_silicone_pad  
        self.left_pad_body = 22      # left_silicone_pad
        
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
    
    def get_end_effector_pose(self, joint_positions):
        """Get the pose of the gripper center (between finger pads)."""
        # Set joint positions and compute forward kinematics
        self.data.qpos[:len(joint_positions)] = joint_positions
        mujoco.mj_forward(self.model, self.data)
        
        # Calculate center between gripper finger pads
        right_pad_pos = self.data.xpos[self.right_pad_body]
        left_pad_pos = self.data.xpos[self.left_pad_body]
        gripper_center = (right_pad_pos + left_pad_pos) / 2
        
        # Use gripper_base orientation (could be improved to average finger orientations)
        gripper_base_rot = self.data.xmat[self.end_effector_body].reshape(3, 3)
        
        return gripper_center, gripper_base_rot

    def forward_kinematics(self, joint_positions):
        """Compute forward kinematics for given joint positions."""
        position, rotation = self.get_end_effector_pose(joint_positions)
        return position, rotation

    def inverse_kinematics(self, target_position, target_orientation=None, 
                          initial_guess=None, tolerance=1e-3, max_iterations=100):
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_position: 3D target position
            target_orientation: 3x3 rotation matrix (optional)
            initial_guess: Initial joint configuration
            tolerance: Position tolerance
            max_iterations: Maximum optimization iterations
        """
        if initial_guess is None:
            initial_guess = np.zeros(7)  # 7 DOF for Kinova Gen3
        
        # Joint limits for Kinova Gen3 (approximate)
        joint_limits = [
            (-2.9, 2.9),    # Joint 1: ±166°
            (-2.1, 2.1),    # Joint 2: ±120°  
            (-2.9, 2.9),    # Joint 3: ±166°
            (-2.1, 2.1),    # Joint 4: ±120°
            (-2.9, 2.9),    # Joint 5: ±166°
            (-2.1, 2.1),    # Joint 6: ±120°
            (-2.9, 2.9),    # Joint 7: ±166°
        ]
        
        def objective(joint_positions):
            try:
                current_position, current_orientation = self.forward_kinematics(joint_positions)
                
                # Position error
                position_error = np.linalg.norm(current_position - target_position)
                
                # Orientation error (if provided)
                orientation_error = 0.0
                if target_orientation is not None:
                    # Use Frobenius norm of rotation matrix difference
                    rot_diff = current_orientation - target_orientation
                    orientation_error = np.linalg.norm(rot_diff) * 0.1  # Weight orientation less
                
                return position_error + orientation_error
                
            except Exception as e:
                return 1e6  # Large penalty for invalid configurations
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=joint_limits,
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )

        print(f"IK result: {result}")
        
        if result.success:
            return result.x, True
        else:
            return initial_guess, False
    
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose"""
        position = self.data.xpos[self.end_effector_body].copy()
        orientation = self.data.xmat[self.end_effector_body].reshape(3, 3).copy()
        return position, orientation
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits for planning"""
        return self.joint_limits_lower.copy(), self.joint_limits_upper.copy() 