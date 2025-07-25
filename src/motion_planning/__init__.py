"""
Motion Planning Package for Kinova Gen3

This package provides:
- Inverse kinematics using MuJoCo
- Path planning with RRT (can be replaced with OMPL)
- Integration with MuJoCo simulation
"""

from .kinematics import KinematicsSolver
from .planners import MotionPlannerFactory
from .utils import Obstacle, PillarObstacle
from .unconstrained_trajopt import UnconstrainedTrajOptPlanner
from .RRTPlanner import RRTPlanner
from .integration import MotionPlanningInterface, TrajectoryVisualizer
import mujoco.viewer

# Import the new abstraction modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scene_builder import MujocoSceneBuilder, create_standard_scene, create_pareto_scene
from trajectory_visualizer import TrajectoryVisualizationManager, MultiTrajectoryVisualizer, create_trajectory_visualizer
from trajectory_optimization_demo import TrajectoryOptimizationDemo, MultiTrajectoryDemo

__all__ = ['KinematicsSolver', 'RRTPlanner', 'UnconstrainedTrajOptPlanner', 'MotionPlannerFactory', 'Obstacle', 'PillarObstacle' ,'MotionPlanningInterface', 'TrajectoryVisualizer', 'MujocoSceneBuilder', 'create_standard_scene', 'create_pareto_scene', 'TrajectoryVisualizationManager', 'MultiTrajectoryVisualizer', 'create_trajectory_visualizer', 'TrajectoryOptimizationDemo', 'MultiTrajectoryDemo']
