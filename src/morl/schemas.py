from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class ObstacleSpec:
    center: tuple[float, float, float]
    radius: float
    safe_distance: float = 0.04

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObstacleSpec":
        center = payload["center"]
        return cls(
            center=(float(center[0]), float(center[1]), float(center[2])),
            radius=float(payload["radius"]),
            safe_distance=float(payload.get("safe_distance", 0.04)),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["center"] = list(self.center)
        return payload

    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    planner_seed: int
    start_config: tuple[float, ...]
    target_position: tuple[float, float, float]
    obstacles: tuple[ObstacleSpec, ...]
    family: str = "base"
    difficulty: str = "medium"
    benchmark_profile: str = "baseline"
    geometry_regime: str = "mixed"
    horizon: int = 25
    dt: float = 0.1

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(payload["task_id"]),
            planner_seed=int(payload["planner_seed"]),
            start_config=tuple(float(value) for value in payload["start_config"]),
            target_position=tuple(float(value) for value in payload["target_position"]),
            obstacles=tuple(ObstacleSpec.from_dict(item) for item in payload["obstacles"]),
            family=str(payload.get("family", "base")),
            difficulty=str(payload.get("difficulty", "medium")),
            benchmark_profile=str(payload.get("benchmark_profile", "baseline")),
            geometry_regime=str(payload.get("geometry_regime", "mixed")),
            horizon=int(payload.get("horizon", 25)),
            dt=float(payload.get("dt", 0.1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "planner_seed": self.planner_seed,
            "start_config": list(self.start_config),
            "target_position": list(self.target_position),
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "family": self.family,
            "difficulty": self.difficulty,
            "benchmark_profile": self.benchmark_profile,
            "geometry_regime": self.geometry_regime,
            "horizon": self.horizon,
            "dt": self.dt,
        }

    def start_array(self) -> np.ndarray:
        return np.asarray(self.start_config, dtype=np.float64)

    def target_array(self) -> np.ndarray:
        return np.asarray(self.target_position, dtype=np.float64)

    def obstacle_dicts(self) -> List[Dict[str, Any]]:
        return [obstacle.to_dict() for obstacle in self.obstacles]
