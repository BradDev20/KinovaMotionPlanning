"""
Kinematics module using MuJoCo for forward/inverse kinematics

Limits are set according to the specifications found here: https://github.com/NVlabs/curobo/blob/main/src/curobo/content/assets/robot/kinova/kinova_gen3_7dof.urdf
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
from typing import Tuple, Optional
from .utils import PerformanceTimer


class KinematicsSolver:
    """Kinematics solver using MuJoCo for the Kinova Gen3 robot"""
    
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # Use gripper center instead of just bracelet_link
        self.end_effector_body = 10  # gripper_base for reference
        self.right_pad_body = 16     # right_silicone_pad  
        self.left_pad_body = 22      # left_silicone_pad
        self.arm_dofs = 7  # 7 DOF for Kinova Gen3
        
        # Joint limits based on Kinova Gen3 specifications
        self.joint_limits_lower = np.array([
            -6.0,  # joint_1
            -2.41,  # joint_2
            -6.0,  # joint_3
            -2.66,  # joint_4
            -6.0,  # joint_5
            -2.23,  # joint_6
            -6.0   # joint_7
        ])

        self.joint_limits_upper = np.array([
            6.0,  # joint_1
            2.41,  # joint_2
            6.0,  # joint_3
            2.66,  # joint_4
            6.0,  # joint_5
            2.23,  # joint_6
            6.0   # joint_7
        ])
        
        # For optimization, use reasonable bounds for unlimited joints
        self.optimization_limits_lower = self.joint_limits_lower.copy()
        self.optimization_limits_upper = self.joint_limits_upper.copy()
        
        # Backup current state
        self._backup_qpos = None
        self._backup_qvel = None
        
        # Performance timer
        self.timer = PerformanceTimer()
    
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
        with self.timer.time_operation('FK'):
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
        with self.timer.time_operation('IK'):
            if initial_guess is None:
                initial_guess = np.zeros(self.arm_dofs)  # 7 DOF for Kinova Gen3
            
            # Joint limits for Kinova Gen3
            bounds = list(zip(self.joint_limits_lower, self.joint_limits_upper))

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
                method='L-BFGS-B',  #TODO: consider using 'SLSQP' (Sequential Least Squares Programming) for constraints
                bounds=bounds,
                options={'maxiter': max_iterations, 'ftol': tolerance}
            )
            # Documentation for SLSQP found here - https://docs.scipy.org/doc/scipy/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp

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
