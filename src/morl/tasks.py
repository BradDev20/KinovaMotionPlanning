from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .schemas import ObstacleSpec, TaskSpec


BASE_START_CONFIG = np.asarray([0.0, -0.5, 0.0, 2.5, 0.0, 1.0, -1.57], dtype=np.float64)
BASE_TARGET_POSITION = np.asarray([0.65, 0.0, 0.2], dtype=np.float64)
WORKSPACE_ANCHOR = np.asarray([0.26, 0.0, 0.24], dtype=np.float64)
JOINT_LIMITS_LOWER = np.asarray([-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14], dtype=np.float64)
JOINT_LIMITS_UPPER = np.asarray([3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14], dtype=np.float64)
MAX_OBSTACLES = 4

LEGACY_FAMILIES = (
    "corridor_left_right",
    "pinch_point",
    "stacked_detour",
    "asymmetric_safe_margin",
)
CONVEX_FAMILIES = (
    "asymmetric_safe_margin",
    "stacked_detour",
)
NONCONVEX_FAMILIES = (
    "pinch_bottleneck",
    "double_corridor",
    "culdesac_escape",
    "offset_gate",
)
MIN_SUCCESSFUL_TRAIN_TASKS_PER_FAMILY = 5
MIN_SUCCESSFUL_VAL_TASKS_PER_FAMILY = 1
MIN_SUCCESSFUL_TEST_TASKS_PER_FAMILY = 1
MIN_SUCCESSFUL_TASKS_PER_FAMILY = (
    MIN_SUCCESSFUL_TRAIN_TASKS_PER_FAMILY + MIN_SUCCESSFUL_VAL_TASKS_PER_FAMILY + MIN_SUCCESSFUL_TEST_TASKS_PER_FAMILY
)

TASK_FAMILY_NAMES = {
    "mixed": "mixed",
    "all": "all",
    "corridor": "corridor_left_right",
    "corridor_left_right": "corridor_left_right",
    "pinch": "pinch_point",
    "pinch_point": "pinch_point",
    "stacked": "stacked_detour",
    "stacked_detour": "stacked_detour",
    "asymmetric": "asymmetric_safe_margin",
    "asymmetric_safe_margin": "asymmetric_safe_margin",
    "pinch_bottleneck": "pinch_bottleneck",
    "double_corridor": "double_corridor",
    "culdesac_escape": "culdesac_escape",
    "offset_gate": "offset_gate",
}


@dataclass(frozen=True)
class TaskFamilyConfig:
    start_noise_scale: tuple[float, ...] = (0.12, 0.12, 0.12, 0.18, 0.12, 0.18, 0.12)
    target_noise_scale: tuple[float, float, float] = (0.03, 0.05, 0.03)
    horizon: int = 50
    dt: float = 0.1
    task_family: str = "mixed"
    difficulty: str = "medium"
    benchmark_profile: str = "baseline"
    geometry_regime: str = "mixed"
    family_mix: tuple[tuple[str, float], ...] | None = None


def normalize_family_name(family_name: str) -> str:
    normalized = str(family_name).strip().lower()
    if normalized not in TASK_FAMILY_NAMES:
        raise ValueError(f"Unsupported task family: {family_name}")
    return TASK_FAMILY_NAMES[normalized]


def parse_family_mix(spec: str | None) -> tuple[tuple[str, float], ...] | None:
    if not spec:
        return None
    weights = []
    for chunk in spec.split(","):
        key, raw_value = chunk.split("=", 1)
        weights.append((normalize_family_name(key.strip()), float(raw_value.strip())))
    total = sum(weight for _, weight in weights)
    if total <= 0.0:
        raise ValueError("Family mix weights must sum to a positive value.")
    return tuple((name, weight / total) for name, weight in weights)


def default_family_mix(benchmark_profile: str, geometry_regime: str) -> tuple[tuple[str, float], ...]:
    profile = str(benchmark_profile).strip().lower()
    regime = str(geometry_regime).strip().lower()

    if regime == "convex":
        if profile == "max_favoring":
            return (
                ("asymmetric_safe_margin", 0.60),
                ("stacked_detour", 0.40),
            )
        return (
            ("asymmetric_safe_margin", 0.50),
            ("stacked_detour", 0.50),
        )

    if regime == "nonconvex":
        if profile == "max_favoring":
            return (
                ("pinch_bottleneck", 0.35),
                ("double_corridor", 0.30),
                ("culdesac_escape", 0.20),
                ("offset_gate", 0.15),
            )
        return (
            ("pinch_bottleneck", 0.25),
            ("double_corridor", 0.25),
            ("culdesac_escape", 0.25),
            ("offset_gate", 0.25),
        )

    if profile == "max_favoring":
        return (
            ("pinch_bottleneck", 0.25),
            ("double_corridor", 0.20),
            ("culdesac_escape", 0.15),
            ("offset_gate", 0.10),
            ("asymmetric_safe_margin", 0.20),
            ("stacked_detour", 0.10),
        )

    return (
        ("corridor_left_right", 0.25),
        ("pinch_point", 0.25),
        ("stacked_detour", 0.25),
        ("asymmetric_safe_margin", 0.25),
    )


def regime_families(benchmark_profile: str, geometry_regime: str) -> tuple[str, ...]:
    return tuple(name for name, _ in default_family_mix(benchmark_profile, geometry_regime))


def generate_alpha_values(alpha_count: int, schedule: str = "linear") -> list[float]:
    count = max(2, int(alpha_count))
    if schedule == "linear":
        values = np.linspace(0.0, 1.0, count)
    elif schedule == "dense-middle":
        values = 0.5 + 0.5 * np.sin(np.linspace(-np.pi / 2, np.pi / 2, count))
    elif schedule == "dense-ends":
        values = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, count)))
    else:
        raise ValueError(f"Unsupported alpha schedule: {schedule}")
    return [float(value) for value in np.clip(values, 0.0, 1.0)]


def split_task_ids(task_ids: Iterable[str], seed: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, list[str]]:
    ids = list(task_ids)
    rng = np.random.default_rng(int(seed))
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    count = len(shuffled)
    train_end = int(round(count * train_ratio))
    val_end = train_end + int(round(count * val_ratio))
    if count >= 3:
        train_end = max(1, min(train_end, count - 2))
        val_end = max(train_end + 1, min(val_end, count - 1))
    else:
        train_end = count
        val_end = count

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def split_successful_task_ids_by_family(
    task_ids_by_family: dict[str, Iterable[str]],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_train_tasks: int = MIN_SUCCESSFUL_TRAIN_TASKS_PER_FAMILY,
    min_val_tasks: int = MIN_SUCCESSFUL_VAL_TASKS_PER_FAMILY,
    min_test_tasks: int = MIN_SUCCESSFUL_TEST_TASKS_PER_FAMILY,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    split_by_family: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}
    merged_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    minimum_total = int(min_train_tasks) + int(min_val_tasks) + int(min_test_tasks)

    for family in sorted(task_ids_by_family.keys()):
        family_task_ids = sorted({str(task_id) for task_id in task_ids_by_family[family]})
        total = len(family_task_ids)
        if total < minimum_total:
            raise ValueError(
                f"Family {family} has only {total} successful tasks; require at least {minimum_total} "
                f"to allocate train={min_train_tasks}, val={min_val_tasks}, test={min_test_tasks}."
            )

        shuffled = family_task_ids.copy()
        rng.shuffle(shuffled)

        train_count = int(round(total * float(train_ratio)))
        val_count = int(round(total * float(val_ratio)))
        train_count = max(int(min_train_tasks), min(train_count, total - int(min_val_tasks) - int(min_test_tasks)))
        val_count = max(int(min_val_tasks), min(val_count, total - train_count - int(min_test_tasks)))
        test_count = total - train_count - val_count
        if test_count < int(min_test_tasks):
            deficit = int(min_test_tasks) - test_count
            reduce_train = min(deficit, max(0, train_count - int(min_train_tasks)))
            train_count -= reduce_train
            deficit -= reduce_train
            reduce_val = min(deficit, max(0, val_count - int(min_val_tasks)))
            val_count -= reduce_val
            deficit -= reduce_val
            if deficit > 0:
                raise ValueError(f"Unable to allocate minimum split floor for family {family}.")
            test_count = total - train_count - val_count

        train_ids = shuffled[:train_count]
        val_ids = shuffled[train_count : train_count + val_count]
        test_ids = shuffled[train_count + val_count :]
        split_by_family["train"][family] = train_ids
        split_by_family["val"][family] = val_ids
        split_by_family["test"][family] = test_ids
        merged_splits["train"].extend(train_ids)
        merged_splits["val"].extend(val_ids)
        merged_splits["test"].extend(test_ids)

    return {
        "splits": merged_splits,
        "split_by_family": split_by_family,
    }


def save_tasks(tasks: Iterable[TaskSpec], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [task.to_dict() for task in tasks]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tasks(path: str | Path) -> list[TaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TaskSpec.from_dict(item) for item in payload]


class TaskSampler:
    def __init__(self, seed: int, family_config: TaskFamilyConfig | None = None):
        self.seed = int(seed)
        self.family_config = family_config or TaskFamilyConfig()
        self._rng = np.random.default_rng(self.seed)
        mix = self.family_config.family_mix or default_family_mix(
            self.family_config.benchmark_profile,
            self.family_config.geometry_regime,
        )
        self._family_names = [name for name, _ in mix]
        self._family_weights = np.asarray([weight for _, weight in mix], dtype=np.float64)
        self._family_weights = self._family_weights / self._family_weights.sum()

    def _difficulty_scale(self) -> dict[str, float]:
        if self.family_config.difficulty == "easy":
            return {"radius": 0.9, "safe": 0.9, "spacing": 1.1, "target": 0.8}
        if self.family_config.difficulty == "hard":
            return {"radius": 1.15, "safe": 1.20, "spacing": 0.82, "target": 1.15}
        return {"radius": 1.0, "safe": 1.0, "spacing": 1.0, "target": 1.0}

    def _sample_start_config(self) -> np.ndarray:
        scale = np.asarray(self.family_config.start_noise_scale, dtype=np.float64)
        sample = BASE_START_CONFIG + self._rng.normal(loc=0.0, scale=scale)
        return np.clip(sample, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    def _sample_target_position(self, family_name: str, difficulty_scale: dict[str, float]) -> np.ndarray:
        scale = np.asarray(self.family_config.target_noise_scale, dtype=np.float64) * difficulty_scale["target"]
        offset = self._rng.uniform(low=-scale, high=scale)
        sample = BASE_TARGET_POSITION + offset
        if family_name in {"stacked_detour", "offset_gate"}:
            sample[2] += self._rng.uniform(-0.02, 0.03)
        if family_name in {"corridor_left_right", "double_corridor", "culdesac_escape"}:
            sample[1] += self._rng.uniform(-0.03, 0.03)
        sample[0] = float(np.clip(sample[0], 0.56, 0.72))
        sample[1] = float(np.clip(sample[1], -0.18, 0.18))
        sample[2] = float(np.clip(sample[2], 0.12, 0.31))
        return sample

    def _make_obstacle(self, center: tuple[float, float, float], radius: float, safe_distance: float) -> ObstacleSpec:
        return ObstacleSpec(
            center=(float(center[0]), float(center[1]), float(center[2])),
            radius=float(radius),
            safe_distance=float(safe_distance),
        )

    def _obstacle_clearance_scales(self, family_name: str) -> tuple[float, float]:
        if family_name in {"asymmetric_safe_margin", "stacked_detour"}:
            radius_scale = 0.82
            safe_scale = 0.72
        elif family_name in {"offset_gate", "double_corridor", "culdesac_escape"}:
            radius_scale = 0.86
            safe_scale = 0.78
        else:
            radius_scale = 0.88
            safe_scale = 0.82

        if self.family_config.difficulty == "easy":
            radius_scale *= 0.94
            safe_scale *= 0.92
        elif self.family_config.difficulty == "hard":
            radius_scale *= 1.04
            safe_scale *= 1.06

        return float(radius_scale), float(safe_scale)

    def _make_family_obstacle(
        self,
        family_name: str,
        center: tuple[float, float, float],
        radius: float,
        safe_distance: float,
    ) -> ObstacleSpec:
        radius_scale, safe_scale = self._obstacle_clearance_scales(family_name)
        return self._make_obstacle(
            center=center,
            radius=radius * radius_scale,
            safe_distance=safe_distance * safe_scale,
        )

    def _clearance_push_distance(self) -> float:
        if self.family_config.difficulty == "easy":
            distance = 0.05
        elif self.family_config.difficulty == "hard":
            distance = 0.033
        else:
            distance = 0.04
        if self.family_config.benchmark_profile == "max_favoring":
            distance += 0.004
        return distance

    def _fallback_push_direction(self, center: np.ndarray, target: np.ndarray, family_name: str, index: int) -> np.ndarray:
        if family_name in {"stacked_detour", "culdesac_escape"}:
            sign = np.sign(center[2] - target[2])
            if abs(float(sign)) < 1e-6:
                sign = 1.0 if (index % 2 == 0) else -1.0
            return np.asarray([0.0, 0.0, float(sign)], dtype=np.float64)

        sign_y = np.sign(center[1])
        if abs(float(sign_y)) < 1e-6:
            sign_y = 1.0 if (index % 2 == 0) else -1.0
        direction = np.asarray([0.0, float(sign_y), 0.0], dtype=np.float64)
        if family_name == "offset_gate":
            direction[2] = 0.35 * (1.0 if center[2] >= target[2] else -1.0)
        return direction / max(float(np.linalg.norm(direction)), 1e-6)

    def _spread_obstacles_from_centerline(
        self,
        target: np.ndarray,
        obstacles: tuple[ObstacleSpec, ...],
        family_name: str,
    ) -> tuple[ObstacleSpec, ...]:
        push_distance = self._clearance_push_distance()
        if push_distance <= 0.0:
            return obstacles

        segment = target - WORKSPACE_ANCHOR
        segment_norm_sq = float(np.dot(segment, segment))
        adjusted: list[ObstacleSpec] = []
        for index, obstacle in enumerate(obstacles):
            center = obstacle.center_array()
            if segment_norm_sq <= 1e-12:
                direction = self._fallback_push_direction(center, target, family_name, index)
            else:
                t = float(np.clip(np.dot(center - WORKSPACE_ANCHOR, segment) / segment_norm_sq, 0.0, 1.0))
                projection = WORKSPACE_ANCHOR + t * segment
                direction = center - projection
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-6:
                    direction = self._fallback_push_direction(center, target, family_name, index)
                else:
                    direction = direction / norm

            shifted = center + push_distance * direction
            shifted[0] = float(np.clip(shifted[0], 0.38, 0.72))
            shifted[1] = float(np.clip(shifted[1], -0.24, 0.24))
            shifted[2] = float(np.clip(shifted[2], 0.10, 0.34))
            adjusted.append(
                ObstacleSpec(
                    center=(float(shifted[0]), float(shifted[1]), float(shifted[2])),
                    radius=obstacle.radius,
                    safe_distance=obstacle.safe_distance,
                )
            )
        return tuple(adjusted)

    def _corridor_left_right(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "corridor_left_right"
        radius = 0.045 * difficulty_scale["radius"]
        safe = 0.045 * difficulty_scale["safe"]
        spacing = 0.11 * difficulty_scale["spacing"]
        base_x = 0.49 + self._rng.uniform(-0.02, 0.02)
        obstacles = []
        for index, y_offset in enumerate((-spacing, 0.0, spacing)):
            x = base_x + 0.04 * index
            y = float(y_offset + self._rng.uniform(-0.012, 0.012))
            z = float(0.2 + self._rng.uniform(-0.015, 0.015))
            obstacles.append(self._make_family_obstacle(family_name, (x, y, z), radius, safe))
        return tuple(obstacles)

    def _pinch_point(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "pinch_point"
        radius = 0.048 * difficulty_scale["radius"]
        safe = 0.045 * difficulty_scale["safe"]
        pinch_gap = 0.075 * difficulty_scale["spacing"]
        base_x = 0.5 + self._rng.uniform(-0.02, 0.02)
        return (
            self._make_family_obstacle(family_name, (base_x - 0.03, -pinch_gap, 0.2), radius, safe),
            self._make_family_obstacle(family_name, (base_x, pinch_gap, 0.2), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.045, 0.0, 0.245), radius * 0.95, safe),
            self._make_family_obstacle(family_name, (base_x + 0.045, 0.0, 0.155), radius * 0.95, safe),
        )

    def _stacked_detour(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "stacked_detour"
        radius = 0.044 * difficulty_scale["radius"]
        safe = 0.045 * difficulty_scale["safe"]
        vertical = 0.065 * difficulty_scale["spacing"]
        base_x = 0.5 + self._rng.uniform(-0.02, 0.02)
        y_anchor = self._rng.uniform(-0.035, 0.035)
        return (
            self._make_family_obstacle(family_name, (base_x - 0.04, y_anchor - 0.05, 0.2 - vertical), radius, safe),
            self._make_family_obstacle(family_name, (base_x, y_anchor, 0.2), radius * 1.05, safe),
            self._make_family_obstacle(family_name, (base_x + 0.04, y_anchor + 0.05, 0.2 + vertical), radius, safe),
        )

    def _asymmetric_safe_margin(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "asymmetric_safe_margin"
        radius = 0.05 * difficulty_scale["radius"]
        safe = 0.05 * difficulty_scale["safe"]
        squeeze = 0.055 * difficulty_scale["spacing"]
        base_x = 0.48 + self._rng.uniform(-0.02, 0.02)
        return (
            self._make_family_obstacle(family_name, (base_x, -0.015, 0.2), radius * 1.08, safe * 1.1),
            self._make_family_obstacle(family_name, (base_x + 0.08, squeeze, 0.22), radius * 0.9, safe),
            self._make_family_obstacle(family_name, (base_x + 0.1, -0.11 * difficulty_scale["spacing"], 0.18), radius * 0.9, safe),
        )

    def _pinch_bottleneck(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "pinch_bottleneck"
        radius = 0.048 * difficulty_scale["radius"]
        safe = 0.052 * difficulty_scale["safe"]
        gap = 0.050 * difficulty_scale["spacing"]
        base_x = 0.48 + self._rng.uniform(-0.015, 0.015)
        return (
            self._make_family_obstacle(family_name, (base_x - 0.02, -gap, 0.2), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.01, gap, 0.2), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.08, 0.095, 0.23), radius * 0.9, safe),
            self._make_family_obstacle(family_name, (base_x + 0.12, 0.12, 0.17), radius * 0.85, safe),
        )

    def _double_corridor(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "double_corridor"
        radius = 0.044 * difficulty_scale["radius"]
        safe = 0.038 * difficulty_scale["safe"]
        side = 0.18 * difficulty_scale["spacing"]
        base_x = 0.47 + self._rng.uniform(-0.015, 0.015)
        return (
            self._make_family_obstacle(family_name, (base_x, 0.0, 0.2), radius * 0.98, safe * 1.05),
            self._make_family_obstacle(family_name, (base_x + 0.045, 0.0, 0.245), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.085, side, 0.2), radius * 0.92, safe),
            self._make_family_obstacle(family_name, (base_x + 0.085, -side, 0.2), radius * 0.92, safe),
        )

    def _culdesac_escape(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "culdesac_escape"
        radius = 0.045 * difficulty_scale["radius"]
        safe = 0.052 * difficulty_scale["safe"]
        base_x = 0.54 + self._rng.uniform(-0.012, 0.012)
        pocket = 0.075 * difficulty_scale["spacing"]
        return (
            self._make_family_obstacle(family_name, (base_x - 0.03, pocket, 0.2), radius, safe),
            self._make_family_obstacle(family_name, (base_x - 0.03, -pocket, 0.2), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.03, 0.0, 0.245), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.03, 0.0, 0.155), radius, safe),
        )

    def _offset_gate(self, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        family_name = "offset_gate"
        radius = 0.043 * difficulty_scale["radius"]
        safe = 0.050 * difficulty_scale["safe"]
        y_high = 0.09 * difficulty_scale["spacing"]
        y_low = 0.03 * difficulty_scale["spacing"]
        base_x = 0.44 + self._rng.uniform(-0.015, 0.015)
        return (
            self._make_family_obstacle(family_name, (base_x, y_high, 0.21), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.05, -y_low, 0.18), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.10, y_low, 0.23), radius, safe),
            self._make_family_obstacle(family_name, (base_x + 0.15, -y_high, 0.19), radius, safe),
        )

    def _family_obstacles(self, family_name: str, difficulty_scale: dict[str, float]) -> tuple[ObstacleSpec, ...]:
        builders = {
            "corridor_left_right": self._corridor_left_right,
            "pinch_point": self._pinch_point,
            "stacked_detour": self._stacked_detour,
            "asymmetric_safe_margin": self._asymmetric_safe_margin,
            "pinch_bottleneck": self._pinch_bottleneck,
            "double_corridor": self._double_corridor,
            "culdesac_escape": self._culdesac_escape,
            "offset_gate": self._offset_gate,
        }
        if family_name not in builders:
            raise ValueError(f"Unsupported normalized family name: {family_name}")
        return builders[family_name](difficulty_scale)

    def _line_distance(self, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
        projection = start + t * segment
        return float(np.linalg.norm(point - projection))

    def _family_valid(self, target: np.ndarray, obstacles: tuple[ObstacleSpec, ...], family_name: str) -> bool:
        if len(obstacles) < 3:
            return False
        obstacle_array = np.asarray([obstacle.center for obstacle in obstacles], dtype=np.float64)
        clearance = np.asarray([obstacle.radius + obstacle.safe_distance for obstacle in obstacles], dtype=np.float64)
        if np.any(np.linalg.norm(obstacle_array - target[None, :], axis=1) < 0.07):
            return False

        y_span = float(obstacle_array[:, 1].max() - obstacle_array[:, 1].min())
        z_span = float(obstacle_array[:, 2].max() - obstacle_array[:, 2].min())
        centerline_distances = np.asarray(
            [self._line_distance(center, WORKSPACE_ANCHOR, target) for center in obstacle_array],
            dtype=np.float64,
        )
        centerline_hits = int(np.sum(centerline_distances <= clearance + 0.015))

        if family_name in {"corridor_left_right", "pinch_point", "stacked_detour", "asymmetric_safe_margin"}:
            return bool(y_span >= 0.05 or z_span >= 0.05)

        side_pos = int(np.sum(obstacle_array[:, 1] > 0.05))
        side_neg = int(np.sum(obstacle_array[:, 1] < -0.05))
        above = int(np.sum(obstacle_array[:, 2] > target[2] + 0.02))
        below = int(np.sum(obstacle_array[:, 2] < target[2] - 0.02))

        if family_name == "pinch_bottleneck":
            return centerline_hits >= 2 and (side_pos >= 2 or side_neg >= 2)
        if family_name == "double_corridor":
            return centerline_hits >= 2 and side_pos >= 1 and side_neg >= 1 and y_span >= 0.12
        if family_name == "culdesac_escape":
            return centerline_hits >= 2 and side_pos >= 1 and side_neg >= 1 and above >= 1 and below >= 1
        if family_name == "offset_gate":
            return centerline_hits >= 1 and side_pos >= 2 and side_neg >= 2 and y_span >= 0.10
        return False

    def _sample_family_name(self) -> str:
        task_family = normalize_family_name(self.family_config.task_family)
        if task_family not in {"mixed", "all"}:
            return task_family
        return str(self._rng.choice(self._family_names, p=self._family_weights))

    def sample_task(self, index: int) -> TaskSpec:
        planner_seed = int(self._rng.integers(0, 2**31 - 1))
        difficulty_scale = self._difficulty_scale()
        family_name = self._sample_family_name()
        target = self._sample_target_position(family_name, difficulty_scale)
        obstacles = self._spread_obstacles_from_centerline(
            target,
            self._family_obstacles(family_name, difficulty_scale),
            family_name,
        )

        attempts = 0
        while not self._family_valid(target, obstacles, family_name) and attempts < 24:
            family_name = self._sample_family_name()
            target = self._sample_target_position(family_name, difficulty_scale)
            obstacles = self._spread_obstacles_from_centerline(
                target,
                self._family_obstacles(family_name, difficulty_scale),
                family_name,
            )
            attempts += 1

        return TaskSpec(
            task_id=f"task_{index:04d}",
            planner_seed=planner_seed,
            start_config=tuple(float(value) for value in self._sample_start_config()),
            target_position=tuple(float(value) for value in target),
            obstacles=obstacles,
            family=family_name,
            difficulty=self.family_config.difficulty,
            benchmark_profile=self.family_config.benchmark_profile,
            geometry_regime=self.family_config.geometry_regime,
            horizon=self.family_config.horizon,
            dt=self.family_config.dt,
        )

    def sample_tasks(self, count: int) -> list[TaskSpec]:
        return [self.sample_task(index) for index in range(int(count))]
