from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TorchPlannerJob:
    worker_id: int
    request_id: str
    order_index: int
    start_config: tuple[float, ...]
    goal_config: tuple[float, ...]
    obstacle_centers: tuple[tuple[float, float, float], ...]
    obstacle_radii: tuple[float, ...]
    obstacle_safe_distances: tuple[float, ...]
    dt: float
    alpha: float
    planner_mode: str
    rho: float
    safety_aggregate: str
    safety_decay_rate: float
    safety_bias: float
    safety_collision_penalty: float
    seed: int


@dataclass(frozen=True)
class TorchPlannerResult:
    worker_id: int
    request_id: str
    order_index: int
    trajectory: np.ndarray
    iterations: int
    final_optimization_cost: float
    scalarized_surrogate_cost: float
    optimizer_steps: int
    batched_jobs: int
    device: str
    duration_sec: float
    surrogate_initial_trajectory_dynamics: dict[str, object]
    surrogate_trajectory_dynamics: dict[str, object]
    surrogate_dynamics_checkpoints: tuple[dict[str, object], ...] = field(default_factory=tuple)
