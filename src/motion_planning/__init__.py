"""
Motion planning package exports.

Keep package imports lightweight so torch-only utilities can be used in
environments that do not have MuJoCo installed.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "KinematicsSolver": ("src.motion_planning.kinematics", "KinematicsSolver"),
    "MotionPlannerFactory": ("src.motion_planning.planners", "MotionPlannerFactory"),
    "Obstacle": ("src.motion_planning.utils", "Obstacle"),
    "PillarObstacle": ("src.motion_planning.utils", "PillarObstacle"),
    "UnconstrainedTrajOptPlanner": ("src.motion_planning.unconstrained_trajopt", "UnconstrainedTrajOptPlanner"),
    "RRTPlanner": ("src.motion_planning.RRTPlanner", "RRTPlanner"),
    "MotionPlanningInterface": ("src.motion_planning.integration", "MotionPlanningInterface"),
    "TrajectoryVisualizer": ("src.motion_planning.integration", "TrajectoryVisualizer"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
