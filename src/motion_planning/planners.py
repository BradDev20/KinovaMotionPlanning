"""
Path planning algorithms for robot motion planning
"""

import numpy as np
import mujoco
from typing import List, Tuple, Callable, Optional, Dict, Any
from scipy.optimize import minimize
from dataclasses import dataclass
from .kinematics import KinematicsSolver
from .utils import Obstacle
from .RRTPlanner import RRTPlanner
from .TrajOpt import TrajOptPlanner


class MotionPlannerFactory:
    """Factory for creating motion planners"""
    
    @staticmethod
    def create_rrt_planner(model: mujoco.MjModel, 
                          data: mujoco.MjData, 
                          **kwargs) -> RRTPlanner:
        """Create an RRT planner"""
        return RRTPlanner(model, data, **kwargs)
    
    @staticmethod
    def create_trajopt_planner(model: mujoco.MjModel,
                              data: mujoco.MjData,
                              **kwargs) -> TrajOptPlanner:
        """Create a TrajOpt planner"""
        return TrajOptPlanner(model, data, **kwargs)
    
    @staticmethod
    def create_collision_checker(model: mujoco.MjModel, 
                               data: mujoco.MjData) -> Callable[[np.ndarray], bool]:
        """Create a collision checking function using MuJoCo"""
        
        # Store original state
        original_qpos = data.qpos.copy()
        original_qvel = data.qvel.copy()
        
        def check_collision(joint_config: np.ndarray) -> bool:
            """Check if joint configuration results in collision"""
            # Set joint configuration
            data.qpos[:7] = joint_config
            data.qvel[:] = 0  # Zero velocities for static check
            
            # Forward dynamics to update contact information
            mujoco.mj_forward(model, data)
            
            # Check for any contacts (collisions)
            has_collision = data.ncon > 0
            
            # Restore original state
            data.qpos[:] = original_qpos
            data.qvel[:] = original_qvel
            mujoco.mj_forward(model, data)
            
            return has_collision
        
        return check_collision 