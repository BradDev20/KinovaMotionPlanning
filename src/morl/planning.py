from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.motion_planning.torch_trajopt import TorchPlannerJob, TorchPlannerResult, TorchTrajectoryBatchPlanner

from .scalarization import alpha_to_weights, scalarize_numpy
from .schemas import TaskSpec


@dataclass(frozen=True)
class PlannerConfig:
    n_waypoints: int = 25
    cost_sample_rate: int = 2
    max_velocity: float = 1.3
    max_acceleration: float = 0.7
    rho: float = 0.01
    warm_start_noise: float = 0.08
    planner_max_iter: int | None = None
    planner_max_fun: int | None = None
    benchmark_profile: str = "baseline"
    geometry_regime: str = "mixed"
    safety_aggregate: str = "avg"
    safety_decay_rate: float = 15.0
    safety_bias: float = -0.08
    safety_collision_penalty: float = 1.0
    device: str = "cpu"
    gpu_batch_size: int = 32
    gpu_batch_timeout_ms: int = 10
    planner_steps: int = 250
    repair_max_iter: int = 40
    repair_max_fun: int = 160


@dataclass
class TaskPlanningContext:
    task: TaskSpec
    scene_path: str
    goal_config: np.ndarray
    model: Any
    data: Any
    kinematics: Any
    length_cost: Any
    safety_cost: Any


def resolve_collection_device(device: str | None = None, *, strict_explicit_cuda: bool = False) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    requested = str(device).strip() if device else ""
    if not requested:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if strict_explicit_cuda:
            raise RuntimeError(
                "CUDA was requested for dataset collection, but the active PyTorch build has no CUDA device available."
            )
        return "cpu"
    return requested


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relative_robot_scene_path(filename: str) -> str:
    return filename


def build_task_scene(task: TaskSpec, scene_dir: str | Path) -> str:
    from src.motion_planning.utils import Obstacle
    from src.motion_planning.scene_builder import create_standard_scene

    output_filename = relative_robot_scene_path(f"morl_{task.task_id}_scene.xml")
    obstacles = [
        Obstacle(center=obstacle.center_array(), radius=obstacle.radius, safe_distance=obstacle.safe_distance)
        for obstacle in task.obstacles
    ]
    return create_standard_scene(
        obstacles=obstacles,
        target_position=task.target_array(),
        trace_dot_count=max(task.horizon * 2, 100),
        output_filename=output_filename,
    )


def _make_warm_start(start_config: np.ndarray, goal_config: np.ndarray, n_waypoints: int, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    trajectory = np.zeros((n_waypoints, start_config.shape[0]), dtype=np.float64)
    for index in range(n_waypoints):
        alpha = index / max(n_waypoints - 1, 1)
        trajectory[index] = (1.0 - alpha) * start_config + alpha * goal_config
    if n_waypoints > 2 and noise_scale > 0.0:
        noise = rng.normal(loc=0.0, scale=noise_scale, size=trajectory[1:-1].shape)
        trajectory[1:-1] += noise
    return trajectory


def _plasma_color(alpha: float) -> list[float]:
    try:
        import matplotlib.pyplot as plt

        rgba = plt.cm.plasma(float(np.clip(alpha, 0.0, 1.0)))
        return [float(rgba[0]), float(rgba[1]), float(rgba[2]), 0.8]
    except ImportError:
        return [float(alpha), float(1.0 - alpha), 0.6, 0.8]


def _trajectory_id(task: TaskSpec, mode: str, alpha: float, restart_index: int) -> str:
    return f"{task.task_id}_{mode}_a{alpha:.3f}_r{restart_index:02d}".replace(".", "p")


def _validate_trajectory_dynamics(
    trajectory: np.ndarray,
    dt: float,
    max_velocity: float,
    max_acceleration: float,
) -> None:
    if trajectory.shape[0] < 2:
        return
    velocities = np.abs(np.diff(trajectory, axis=0) / float(dt))
    max_velocity_allowed = float(max_velocity) + 1e-6
    velocity_violations = int(np.sum(velocities > max_velocity_allowed))
    if velocity_violations > 0:
        max_velocity_observed = float(np.max(velocities))
        raise RuntimeError(
            "Planned trajectory exceeds the configured joint velocity limit "
            f"({velocity_violations} violations, max velocity={max_velocity_observed:.4f}, "
            f"limit={max_velocity_allowed:.4f})."
        )
    if trajectory.shape[0] < 3:
        return
    accelerations = np.abs(np.diff(trajectory, n=2, axis=0) / float(dt ** 2))
    max_acceleration_allowed = float(max_acceleration) + 1e-6
    acceleration_violations = int(np.sum(accelerations > max_acceleration_allowed))
    if acceleration_violations > 0:
        max_acceleration_observed = float(np.max(accelerations))
        raise RuntimeError(
            "Planned trajectory exceeds the configured joint acceleration limit "
            f"({acceleration_violations} violations, max acceleration={max_acceleration_observed:.4f}, "
            f"limit={max_acceleration_allowed:.4f})."
        )


def _trajectory_dynamics_summary(
    trajectory: np.ndarray,
    dt: float,
    max_velocity: float,
    max_acceleration: float,
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "velocity_violation_count": 0,
        "acceleration_violation_count": 0,
        "max_velocity_observed": 0.0,
        "max_acceleration_observed": 0.0,
        "max_velocity_limit": float(max_velocity) + 1e-6,
        "max_acceleration_limit": float(max_acceleration) + 1e-6,
        "max_velocity_excess": 0.0,
        "max_acceleration_excess": 0.0,
    }
    if trajectory.shape[0] >= 2:
        velocities = np.abs(np.diff(trajectory, axis=0) / float(dt))
        if velocities.size > 0:
            max_velocity_observed = float(np.max(velocities))
            velocity_limit = float(summary["max_velocity_limit"])
            summary["max_velocity_observed"] = max_velocity_observed
            summary["velocity_violation_count"] = int(np.sum(velocities > velocity_limit))
            summary["max_velocity_excess"] = max(0.0, max_velocity_observed - velocity_limit)
    if trajectory.shape[0] >= 3:
        accelerations = np.abs(np.diff(trajectory, n=2, axis=0) / float(dt ** 2))
        if accelerations.size > 0:
            max_acceleration_observed = float(np.max(accelerations))
            acceleration_limit = float(summary["max_acceleration_limit"])
            summary["max_acceleration_observed"] = max_acceleration_observed
            summary["acceleration_violation_count"] = int(np.sum(accelerations > acceleration_limit))
            summary["max_acceleration_excess"] = max(0.0, max_acceleration_observed - acceleration_limit)
    return summary


def _validate_trajectory_contacts(model, data, trajectory: np.ndarray) -> None:
    from src.motion_planning.planners import MotionPlannerFactory

    collision_checker = MotionPlannerFactory.create_collision_checker(model, data)
    for waypoint in trajectory:
        result = collision_checker.evaluate(np.asarray(waypoint, dtype=np.float64))
        if result.has_collision and result.first_disallowed is not None:
            contact = result.first_disallowed
            raise RuntimeError(
                "Planned trajectory contains a disallowed MuJoCo contact "
                f"({contact.classification}): "
                f"geom1={contact.geom1}, geom2={contact.geom2}, "
                f"body1={contact.body1}, body2={contact.body2}."
            )


def prepare_task_planning_context(
    task: TaskSpec,
    *,
    scene_dir: str | Path,
    planner_config: PlannerConfig | None = None,
) -> TaskPlanningContext:
    import mujoco

    from src.motion_planning.cost_functions import MuJoCoRobotObstacleCost, TrajectoryLengthCostFunction
    from src.motion_planning.kinematics import KinematicsSolver

    config = planner_config or PlannerConfig()
    scene_path = build_task_scene(task, scene_dir)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    kinematics = KinematicsSolver(scene_path)

    goal_config, ik_success = kinematics.inverse_kinematics(
        task.target_array(),
        initial_guess=task.start_array(),
        tolerance=1e-3,
        max_iterations=2000,
    )
    if not ik_success:
        raise RuntimeError(f"Inverse kinematics failed for {task.task_id}.")

    length_cost = TrajectoryLengthCostFunction(kinematics_solver=kinematics, weight=1.0, normalization_bounds=(0.0, 1.0))
    safety_cost = MuJoCoRobotObstacleCost(
        model,
        data,
        weight=1.0,
        aggregate=config.safety_aggregate,
        collision_penalty=config.safety_collision_penalty,
    )
    return TaskPlanningContext(
        task=task,
        scene_path=scene_path,
        goal_config=np.asarray(goal_config, dtype=np.float64),
        model=model,
        data=data,
        kinematics=kinematics,
        length_cost=length_cost,
        safety_cost=safety_cost,
    )


def build_torch_planner_job(
    context: TaskPlanningContext,
    *,
    alpha: float,
    mode: str,
    restart_index: int,
    order_index: int,
    worker_id: int = -1,
    planner_config: PlannerConfig | None = None,
) -> TorchPlannerJob:
    config = planner_config or PlannerConfig()
    return TorchPlannerJob(
        worker_id=int(worker_id),
        request_id=_trajectory_id(context.task, mode, alpha, restart_index),
        order_index=int(order_index),
        start_config=tuple(float(value) for value in context.task.start_config),
        goal_config=tuple(float(value) for value in context.goal_config),
        obstacle_centers=tuple(tuple(float(axis) for axis in obstacle.center) for obstacle in context.task.obstacles),
        obstacle_radii=tuple(float(obstacle.radius) for obstacle in context.task.obstacles),
        obstacle_safe_distances=tuple(float(obstacle.safe_distance) for obstacle in context.task.obstacles),
        dt=float(context.task.dt),
        alpha=float(alpha),
        planner_mode=str(mode),
        rho=float(config.rho),
        safety_aggregate=str(config.safety_aggregate),
        safety_decay_rate=float(config.safety_decay_rate),
        safety_bias=float(config.safety_bias),
        safety_collision_penalty=float(config.safety_collision_penalty),
        seed=int(context.task.planner_seed + restart_index),
        task_family=str(getattr(context.task, "family", "base")),
    )


def run_torch_planner_batch(
    jobs: list[TorchPlannerJob],
    *,
    planner_config: PlannerConfig | None = None,
) -> list[TorchPlannerResult]:
    config = planner_config or PlannerConfig()
    if not jobs:
        return []
    planner = TorchTrajectoryBatchPlanner(
        device=resolve_collection_device(config.device),
        n_waypoints=config.n_waypoints,
        planner_steps=config.planner_steps,
        max_velocity=config.max_velocity,
        max_acceleration=config.max_acceleration,
        warm_start_noise=config.warm_start_noise,
    )
    return planner.solve_batch(jobs)


def _run_cpu_repair(
    context: TaskPlanningContext,
    *,
    candidate_trajectory: np.ndarray,
    alpha: float,
    mode: str,
    planner_config: PlannerConfig | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    from src.motion_planning.fast_trajopt import FastTrajOptPlanner

    config = planner_config or PlannerConfig()
    weights = alpha_to_weights(alpha)
    effective_dt = dt if dt is not None else context.task.dt
    planner = FastTrajOptPlanner(
        context.model,
        context.data,
        n_waypoints=config.n_waypoints,
        dt=effective_dt,
        max_velocity=config.max_velocity,
        max_acceleration=config.max_acceleration,
        cost_mode="composite",
        cost_sample_rate=config.cost_sample_rate,
        use_global_fk_cache=True,
    )
    planner.max_iter = int(config.repair_max_iter)
    planner.max_fun = int(config.repair_max_fun)
    planner.setup_composite_cost(
        cost_functions=[context.length_cost, context.safety_cost],
        weights=list(weights),
        formulation=mode,
        rho=config.rho,
    )
    repaired, success, metadata = planner.plan(
        context.task.start_array(),
        context.goal_config,
        warm_start_trajectory=np.asarray(candidate_trajectory, dtype=np.float64),
    )
    repaired_array = np.asarray(repaired, dtype=np.float64)
    if repaired_array.size == 0:
        raise RuntimeError("CPU repair returned an empty trajectory.")
    return repaired_array, metadata, bool(success)


def finalize_planned_trajectory(
    context: TaskPlanningContext,
    *,
    planner_result: TorchPlannerResult,
    alpha: float,
    mode: str,
    restart_index: int,
    planner_config: PlannerConfig | None = None,
) -> dict[str, Any]:
    config = planner_config or PlannerConfig()
    weights = alpha_to_weights(alpha)
    trajectory_array = np.asarray(planner_result.trajectory, dtype=np.float64)
    
    if hasattr(planner_result, "dt") and planner_result.dt is not None:
        effective_dt = float(planner_result.dt)
    else:
        effective_dt = float(context.task.dt)
        
    optimization_metadata: dict[str, Any] = {
        "iterations": int(planner_result.iterations),
        "final_optimization_cost": float(planner_result.final_optimization_cost),
        "backend": "torch",
        "device": str(planner_result.device),
        "optimizer_steps": int(planner_result.optimizer_steps),
        "batched_jobs": int(planner_result.batched_jobs),
        "scalarized_surrogate_cost": float(planner_result.scalarized_surrogate_cost),
        "duration_sec": float(planner_result.duration_sec),
        "repair_used": False,
        "repair_validation_failure_reason": None,
        "repair_validation_failure_message": None,
        "validation_passed": False,
    }
    if isinstance(planner_result.surrogate_trajectory_dynamics, dict):
        optimization_metadata["surrogate_trajectory_dynamics"] = dict(planner_result.surrogate_trajectory_dynamics)
    if isinstance(planner_result.surrogate_initial_trajectory_dynamics, dict):
        optimization_metadata["surrogate_initial_trajectory_dynamics"] = dict(
            planner_result.surrogate_initial_trajectory_dynamics
        )
    if isinstance(planner_result.surrogate_dynamics_checkpoints, (list, tuple)):
        optimization_metadata["surrogate_dynamics_checkpoints"] = [
            dict(item) for item in planner_result.surrogate_dynamics_checkpoints if isinstance(item, dict)
        ]
    warm_start_metadata = getattr(planner_result, "warm_start_metadata", None)
    if isinstance(warm_start_metadata, dict):
        optimization_metadata["warm_start_metadata"] = dict(warm_start_metadata)

    validation_error: Exception | None = None
    validation_failure_reason: str | None = None
    raw_dynamics_summary: dict[str, float | int] | None = None
    try:
        _validate_trajectory_dynamics(
            trajectory_array,
            dt=effective_dt,
            max_velocity=config.max_velocity,
            max_acceleration=config.max_acceleration,
        )
    except Exception as exc:
        validation_error = exc
        validation_failure_reason = "dynamics"
        raw_dynamics_summary = _trajectory_dynamics_summary(
            trajectory_array,
            dt=effective_dt,
            max_velocity=config.max_velocity,
            max_acceleration=config.max_acceleration,
        )

    if validation_error is None:
        try:
            _validate_trajectory_contacts(context.model, context.data, trajectory_array)
        except Exception as exc:
            validation_error = exc
            validation_failure_reason = "contacts"

    if validation_error is not None:
        repaired_array, repair_metadata, repair_success = _run_cpu_repair(
            context,
            candidate_trajectory=trajectory_array,
            alpha=alpha,
            mode=mode,
            planner_config=config,
            dt=effective_dt,
        )
        optimization_metadata["repair_used"] = True
        optimization_metadata["repair_validation_failure_reason"] = validation_failure_reason
        optimization_metadata["repair_validation_failure_message"] = str(validation_error)
        if raw_dynamics_summary is not None:
            optimization_metadata["raw_dynamics_violation"] = raw_dynamics_summary
        optimization_metadata["repair"] = {
            **repair_metadata,
            "success": bool(repair_success),
            "validation_failure_reason": validation_failure_reason,
            "validation_failure_message": str(validation_error),
        }
        if raw_dynamics_summary is not None:
            optimization_metadata["repair"]["raw_dynamics_violation"] = raw_dynamics_summary
        try:
            _validate_trajectory_dynamics(
                repaired_array,
                dt=effective_dt,
                max_velocity=config.max_velocity,
                max_acceleration=config.max_acceleration,
            )
            _validate_trajectory_contacts(context.model, context.data, repaired_array)
        except Exception as repair_error:
            raise RuntimeError(
                f"GPU surrogate validation failed ({validation_error}); CPU repair failed ({repair_error})."
            ) from repair_error
        trajectory_array = repaired_array

    optimization_metadata["validation_passed"] = True
    length_value = float(context.length_cost.compute_cost(trajectory_array, effective_dt))
    obstacle_value = float(context.safety_cost.compute_cost(trajectory_array, effective_dt))
    scalarized_value = float(
        scalarize_numpy(
            np.asarray([[length_value, obstacle_value]], dtype=np.float32),
            weights,
            mode,
            rho=config.rho,
        )[0]
    )

    return {
        "trajectory_id": _trajectory_id(context.task, mode, alpha, restart_index),
        "trajectory": trajectory_array,
        "dt": effective_dt,
        "alpha": float(alpha),
        "length_weight": float(weights[0]),
        "obstacle_weight": float(weights[1]),
        "length_cost": length_value,
        "obstacle_cost": obstacle_value,
        "scalarized_cost": scalarized_value,
        "waypoint_count": int(trajectory_array.shape[0]),
        "planner_mode": mode,
        "restart_index": int(restart_index),
        "color": _plasma_color(alpha),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "task_spec": context.task.to_dict(),
        "optimization": optimization_metadata,
        "scene_path": context.scene_path,
        "benchmark_profile": config.benchmark_profile,
        "geometry_regime": config.geometry_regime,
        "safety_aggregate": config.safety_aggregate,
    }


def plan_task_trajectory(
    task: TaskSpec,
    alpha: float,
    mode: str,
    scene_dir: str | Path,
    restart_index: int = 0,
    planner_config: PlannerConfig | None = None,
) -> dict[str, Any]:
    config = planner_config or PlannerConfig()
    context = prepare_task_planning_context(task, scene_dir=scene_dir, planner_config=config)
    job = build_torch_planner_job(
        context,
        alpha=alpha,
        mode=mode,
        restart_index=restart_index,
        order_index=0,
        planner_config=config,
    )
    result = run_torch_planner_batch([job], planner_config=config)[0]
    return finalize_planned_trajectory(
        context,
        planner_result=result,
        alpha=alpha,
        mode=mode,
        restart_index=restart_index,
        planner_config=config,
    )


def save_experiment_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
