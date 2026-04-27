from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from src.numba_compat import numba_njit

from .config import EnvConfig
from .schemas import TaskSpec
from .scalarization import hypervolume_2d, pareto_front
from .tasks import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, NONCONVEX_FAMILIES, load_tasks

if TYPE_CHECKING:
    pass


MODE_TO_INDEX = {"sum": 0, "max": 1}


@numba_njit(cache=True)
def _resample_indices(length: int, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.empty(0, dtype=np.int32)
    indices = np.empty(target_length, dtype=np.int32)
    if target_length == 1:
        indices[0] = 0
        return indices
    if length <= 1:
        for i in range(target_length):
            indices[i] = 0
        return indices
    scale = float(length - 1) / float(target_length - 1)
    for i in range(target_length):
        indices[i] = int(scale * i)
    return indices


@numba_njit(cache=True)
def _trajectory_distance_kernel(lhs_sample: np.ndarray, rhs_sample: np.ndarray) -> float:
    total = 0.0
    rows = lhs_sample.shape[0]
    if rows == 0:
        return 0.0
    cols = lhs_sample.shape[1]
    for i in range(rows):
        norm_sq = 0.0
        for j in range(cols):
            delta = float(lhs_sample[i, j]) - float(rhs_sample[i, j])
            norm_sq += delta * delta
        total += np.sqrt(norm_sq)
    return total / float(rows)


@numba_njit(cache=True)
def _nearest_neighbor_spacing_kernel(points: np.ndarray) -> float:
    count = points.shape[0]
    if count <= 1:
        return 0.0
    total = 0.0
    for i in range(count):
        best = np.inf
        for j in range(count):
            if i == j:
                continue
            dx = float(points[j, 0]) - float(points[i, 0])
            dy = float(points[j, 1]) - float(points[i, 1])
            distance = np.sqrt(dx * dx + dy * dy)
            if distance < best:
                best = distance
        total += best
    return total / float(count)


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _resample_trajectory(trajectory: np.ndarray, target_length: int) -> np.ndarray:
    if len(trajectory) == target_length:
        return trajectory
    indices = _resample_indices(len(trajectory), target_length)
    return trajectory[indices]


def trajectory_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    target_length = min(len(lhs), len(rhs))
    lhs_sample = _resample_trajectory(lhs, target_length)
    rhs_sample = _resample_trajectory(rhs, target_length)
    return float(_trajectory_distance_kernel(lhs_sample, rhs_sample))


def deduplicate_records(
    records: Iterable[dict[str, Any]],
    objective_tol: float = 1e-3,
    path_tol: float = 5e-2,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task_id = str(record["task_spec"]["task_id"])
        grouped.setdefault(task_id, []).append(record)

    unique: list[dict[str, Any]] = []
    for task_id in sorted(grouped.keys()):
        task_records = grouped[task_id]
        task_unique: list[dict[str, Any]] = []
        for record in task_records:
            duplicate = False
            for candidate in task_unique:
                objective_gap = max(
                    abs(float(record["length_cost"]) - float(candidate["length_cost"])),
                    abs(float(record["obstacle_cost"]) - float(candidate["obstacle_cost"])),
                )
                path_gap = trajectory_distance(np.asarray(record["trajectory"]), np.asarray(candidate["trajectory"]))
                if objective_gap <= objective_tol and path_gap <= path_tol:
                    duplicate = True
                    break
            if not duplicate:
                task_unique.append(record)
        unique.extend(task_unique)
    return unique


def cluster_records_by_objective(records: Iterable[dict[str, Any]], objective_tol: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for record in records:
        point = np.asarray([record["length_cost"], record["obstacle_cost"]], dtype=np.float64)
        assigned = False
        for cluster in clusters:
            representative = np.asarray([cluster[0]["length_cost"], cluster[0]["obstacle_cost"]], dtype=np.float64)
            if np.max(np.abs(point - representative)) <= objective_tol:
                cluster.append(record)
                assigned = True
                break
        if not assigned:
            clusters.append([record])
    return clusters


def cluster_records_by_route(records: Iterable[dict[str, Any]], route_tol: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for record in records:
        assigned = False
        trajectory = np.asarray(record["trajectory"], dtype=np.float32)
        for cluster in clusters:
            representative = np.asarray(cluster[0]["trajectory"], dtype=np.float32)
            if trajectory_distance(trajectory, representative) <= route_tol:
                cluster.append(record)
                assigned = True
                break
        if not assigned:
            clusters.append([record])
    return clusters


def _nearest_neighbor_spacing(points: np.ndarray) -> float:
    return float(_nearest_neighbor_spacing_kernel(np.asarray(points, dtype=np.float64)))


def coverage_metrics(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    records_list = list(records)
    if not records_list:
        return {"pareto_count": 0, "hypervolume": 0.0, "front_spacing": 0.0}
    points = np.asarray([[record["length_cost"], record["obstacle_cost"]] for record in records_list], dtype=np.float64)
    front = pareto_front(points)
    reference = (
        float(points[:, 0].max() * 1.05 + 1e-6),
        float(points[:, 1].max() * 1.05 + 1e-6),
    )
    return {
        "pareto_count": int(front.shape[0]),
        "hypervolume": float(hypervolume_2d(points, reference)),
        "front_spacing": _nearest_neighbor_spacing(front),
    }


def family_breakdown(records: Iterable[dict[str, Any]], objective_tol: float, route_tol: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        family_name = str(record["task_spec"].get("family", "base"))
        grouped.setdefault(family_name, []).append(record)
    breakdown = {}
    for family_name, family_records in sorted(grouped.items()):
        objective_clusters = cluster_records_by_objective(family_records, objective_tol=objective_tol)
        route_clusters = cluster_records_by_route(family_records, route_tol=route_tol)
        breakdown[family_name] = {
            "count": len(family_records),
            "unique_objective_count": len(objective_clusters),
            "unique_route_count": len(route_clusters),
            **coverage_metrics(family_records),
        }
    return breakdown


def threshold_sensitivity(records: Iterable[dict[str, Any]], objective_tol: float, path_tol: float) -> dict[str, dict[str, Any]]:
    records_list = list(records)
    sensitivity = {}
    for scale in (0.5, 1.0, 2.0):
        scaled_records = deduplicate_records(
            records_list,
            objective_tol=objective_tol * scale,
            path_tol=path_tol * scale,
        )
        sensitivity[f"x{scale:.1f}"] = {
            "unique_trajectory_count": len(scaled_records),
            **coverage_metrics(scaled_records),
        }
    return sensitivity


def summarize_records(
    records: Iterable[dict[str, Any]],
    objective_tol: float,
    path_tol: float,
    route_tol: float | None = None,
) -> dict[str, Any]:
    records_list = list(records)
    effective_route_tol = float(route_tol if route_tol is not None else path_tol * 2.0)
    objective_clusters = cluster_records_by_objective(records_list, objective_tol=objective_tol)
    route_clusters = cluster_records_by_route(records_list, route_tol=effective_route_tol)
    nonconvex_records = [
        record for record in records_list if str(record["task_spec"].get("family", "base")) in NONCONVEX_FAMILIES
    ]
    nonconvex_route_count = len(cluster_records_by_route(nonconvex_records, route_tol=effective_route_tol)) if nonconvex_records else 0
    return {
        "unique_trajectory_count": len(records_list),
        "unique_objective_count": len(objective_clusters),
        "unique_route_count": len(route_clusters),
        "nonconvex_route_count": nonconvex_route_count,
        "family_breakdown": family_breakdown(records_list, objective_tol=objective_tol, route_tol=effective_route_tol),
        "threshold_sensitivity": threshold_sensitivity(records_list, objective_tol=objective_tol, path_tol=path_tol),
        **coverage_metrics(records_list),
    }


def save_raw_record(record: dict[str, Any], raw_dir: str | Path) -> Path:
    raw_path = ensure_dir(raw_dir) / f"{record['trajectory_id']}.pkl"
    with raw_path.open("wb") as handle:
        pickle.dump(record, handle)
    return raw_path


def save_metadata(records: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    payload = []
    for record in records:
        payload.append(
            {
                "trajectory_id": record["trajectory_id"],
                "task_id": record["task_spec"]["task_id"],
                "planner_mode": record["planner_mode"],
                "alpha": record["alpha"],
                "length_cost": record["length_cost"],
                "obstacle_cost": record["obstacle_cost"],
                "scalarized_cost": record["scalarized_cost"],
                "family": record["task_spec"].get("family", "base"),
                "difficulty": record["task_spec"].get("difficulty", "medium"),
                "benchmark_profile": record["task_spec"].get("benchmark_profile", record.get("benchmark_profile", "baseline")),
                "geometry_regime": record["task_spec"].get("geometry_regime", record.get("geometry_regime", "mixed")),
                "safety_aggregate": record.get("safety_aggregate", "avg"),
                "planner_seed": record["task_spec"].get("planner_seed"),
                "waypoint_count": record["waypoint_count"],
                "color": record["color"],
                "restart_index": record["restart_index"],
                "timestamp": record["timestamp"],
                "filename": f"{record['trajectory_id']}.pkl",
            }
        )
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_raw_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "trajectory_metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = []
    for item in metadata:
        record_path = dataset_path / "raw" / item["filename"]
        if record_path.exists():
            with record_path.open("rb") as handle:
                records.append(pickle.load(handle))
    return records


def record_to_transition_arrays(
    record: dict[str, Any],
    scene_dir: str | Path,
    env_config: "EnvConfig | None" = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    from .env import KinovaMORLEnv

    task = TaskSpec.from_dict(record["task_spec"])
    weights = np.asarray([record["length_weight"], record["obstacle_weight"]], dtype=np.float32)
    trajectory = np.asarray(record["trajectory"], dtype=np.float32)
    required_steps = max(int(task.horizon), max(int(trajectory.shape[0]) - 1, 1))
    if env_config is None:
        effective_env_config = EnvConfig(max_steps=required_steps)
    else:
        configured_steps = int(env_config.max_steps) if env_config.max_steps is not None else 0
        effective_env_config = replace(env_config, max_steps=max(configured_steps, required_steps))
    env = KinovaMORLEnv(task, scene_dir=scene_dir, env_config=effective_env_config)

    observations = []
    actions = []
    next_observations = []
    reward_vectors = []
    objective_vectors = []
    dones = []
    task_ids = []
    trajectory_ids = []
    planner_modes = []
    alphas = []
    weight_vectors = []
    transition_replay_step_count = 0
    action_clip_step_count = 0
    joint_limit_hit_step_count = 0
    constraint_hit_step_count = 0
    max_abs_planned_action = 0.0
    max_abs_executed_action = 0.0
    step_eps = 1e-6
    limit_eps = 1e-5

    observation, _ = env.reset(start_config=trajectory[0])
    observation_dim = int(observation.shape[0])
    action_dim = int(trajectory.shape[1])
    action_limit = float(env._effective_action_scale())
    for index in range(len(trajectory) - 1):
        planned_action = trajectory[index + 1] - trajectory[index]
        clipped_action = np.clip(planned_action, -action_limit, action_limit).astype(np.float32)
        previous_qpos = env.qpos.copy()
        next_observation, reward_vector, objective_vector, done, _ = env.step(planned_action, clip_action=True)
        executed_action = (env.qpos - previous_qpos).astype(np.float32)
        transition_replay_step_count += 1
        max_abs_planned_action = max(max_abs_planned_action, float(np.max(np.abs(planned_action))))
        max_abs_executed_action = max(max_abs_executed_action, float(np.max(np.abs(executed_action))))
        action_was_clipped = bool(np.any(np.abs(planned_action - clipped_action) > step_eps))
        if action_was_clipped:
            action_clip_step_count += 1
        joint_limit_hit = bool(np.any(np.abs(executed_action - clipped_action) > step_eps))
        if not joint_limit_hit:
            lower_hit = np.abs(env.qpos - JOINT_LIMITS_LOWER) <= limit_eps
            upper_hit = np.abs(env.qpos - JOINT_LIMITS_UPPER) <= limit_eps
            attempted_past_lower = clipped_action < -step_eps
            attempted_past_upper = clipped_action > step_eps
            joint_limit_hit = bool(np.any((lower_hit & attempted_past_lower) | (upper_hit & attempted_past_upper)))
        if joint_limit_hit:
            joint_limit_hit_step_count += 1
        if action_was_clipped or joint_limit_hit:
            constraint_hit_step_count += 1
        is_terminal = bool(done or index == len(trajectory) - 2)
        observations.append(observation.copy())
        actions.append(executed_action.copy())
        next_observations.append(next_observation.copy())
        reward_vectors.append(reward_vector.copy())
        objective_vectors.append(objective_vector.copy())
        dones.append(is_terminal)
        task_ids.append(task.task_id)
        trajectory_ids.append(record["trajectory_id"])
        planner_modes.append(record["planner_mode"])
        alphas.append(record["alpha"])
        weight_vectors.append(weights.copy())
        observation = next_observation
        if is_terminal:
            break

    denominator = float(transition_replay_step_count) if transition_replay_step_count > 0 else 1.0
    return (
        {
            "observations": np.asarray(observations, dtype=np.float32).reshape(-1, observation_dim),
            "actions": np.asarray(actions, dtype=np.float32).reshape(-1, action_dim),
            "next_observations": np.asarray(next_observations, dtype=np.float32).reshape(-1, observation_dim),
            "reward_vectors": np.asarray(reward_vectors, dtype=np.float32).reshape(-1, 2),
            "objective_vectors": np.asarray(objective_vectors, dtype=np.float32).reshape(-1, 2),
            "dones": np.asarray(dones, dtype=np.bool_),
            "task_ids": np.asarray(task_ids),
            "trajectory_ids": np.asarray(trajectory_ids),
            "planner_modes": np.asarray(planner_modes),
            "alphas": np.asarray(alphas, dtype=np.float32),
            "weights": np.asarray(weight_vectors, dtype=np.float32),
        },
        {
            "transition_replay_step_count": int(transition_replay_step_count),
            "action_clip_step_count": int(action_clip_step_count),
            "joint_limit_hit_step_count": int(joint_limit_hit_step_count),
            "constraint_hit_step_count": int(constraint_hit_step_count),
            "action_clip_rate": float(action_clip_step_count) / denominator,
            "joint_limit_hit_rate": float(joint_limit_hit_step_count) / denominator,
            "constraint_hit_rate": float(constraint_hit_step_count) / denominator,
            "max_abs_planned_action": float(max_abs_planned_action),
            "max_abs_executed_action": float(max_abs_executed_action),
        },
    )


def _empty_transition_arrays() -> dict[str, np.ndarray]:
    return {
        "observations": np.zeros((0, 0), dtype=np.float32),
        "actions": np.zeros((0, 0), dtype=np.float32),
        "next_observations": np.zeros((0, 0), dtype=np.float32),
        "reward_vectors": np.zeros((0, 2), dtype=np.float32),
        "objective_vectors": np.zeros((0, 2), dtype=np.float32),
        "dones": np.zeros((0,), dtype=np.bool_),
        "task_ids": np.asarray([], dtype="<U1"),
        "trajectory_ids": np.asarray([], dtype="<U1"),
        "planner_modes": np.asarray([], dtype="<U1"),
        "alphas": np.zeros((0,), dtype=np.float32),
        "weights": np.zeros((0, 2), dtype=np.float32),
    }


def build_transition_dataset(
    records: Iterable[dict[str, Any]],
    scene_dir: str | Path,
    output_path: str | Path,
    env_config: "EnvConfig | None" = None,
) -> dict[str, Any]:
    records_list = list(records)
    if not records_list:
        merged = _empty_transition_arrays()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **merged)
        return {
            "transition_count": 0,
            "observation_dim": 0,
            "action_dim": 0,
            "trajectory_count": 0,
        }

    arrays: dict[str, list[np.ndarray]] = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "reward_vectors": [],
        "objective_vectors": [],
        "dones": [],
        "task_ids": [],
        "trajectory_ids": [],
        "planner_modes": [],
        "alphas": [],
        "weights": [],
    }
    diagnostics_totals: dict[str, float] = {
        "transition_replay_step_count": 0.0,
        "action_clip_step_count": 0.0,
        "joint_limit_hit_step_count": 0.0,
        "constraint_hit_step_count": 0.0,
        "max_abs_planned_action": 0.0,
        "max_abs_executed_action": 0.0,
    }

    for record in records_list:
        record_arrays, record_diagnostics = record_to_transition_arrays(record, scene_dir=scene_dir, env_config=env_config)
        for key, value in record_arrays.items():
            arrays[key].append(value)
        diagnostics_totals["transition_replay_step_count"] += float(record_diagnostics["transition_replay_step_count"])
        diagnostics_totals["action_clip_step_count"] += float(record_diagnostics["action_clip_step_count"])
        diagnostics_totals["joint_limit_hit_step_count"] += float(record_diagnostics["joint_limit_hit_step_count"])
        diagnostics_totals["constraint_hit_step_count"] += float(record_diagnostics["constraint_hit_step_count"])
        diagnostics_totals["max_abs_planned_action"] = max(
            diagnostics_totals["max_abs_planned_action"],
            float(record_diagnostics["max_abs_planned_action"]),
        )
        diagnostics_totals["max_abs_executed_action"] = max(
            diagnostics_totals["max_abs_executed_action"],
            float(record_diagnostics["max_abs_executed_action"]),
        )

    merged = {}
    for key, parts in arrays.items():
        merged[key] = np.concatenate(parts, axis=0) if parts else _empty_transition_arrays()[key]

    transition_replay_step_count = int(diagnostics_totals["transition_replay_step_count"])
    denominator = float(transition_replay_step_count) if transition_replay_step_count > 0 else 1.0
    action_clip_step_count = int(diagnostics_totals["action_clip_step_count"])
    joint_limit_hit_step_count = int(diagnostics_totals["joint_limit_hit_step_count"])
    constraint_hit_step_count = int(diagnostics_totals["constraint_hit_step_count"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return {
        "transition_count": int(merged["observations"].shape[0]),
        "observation_dim": int(merged["observations"].shape[1]),
        "action_dim": int(merged["actions"].shape[1]),
        "trajectory_count": len(records_list),
        "transition_replay_step_count": transition_replay_step_count,
        "action_clip_step_count": action_clip_step_count,
        "action_clip_rate": float(action_clip_step_count) / denominator,
        "joint_limit_hit_step_count": joint_limit_hit_step_count,
        "joint_limit_hit_rate": float(joint_limit_hit_step_count) / denominator,
        "constraint_hit_step_count": constraint_hit_step_count,
        "constraint_hit_rate": float(constraint_hit_step_count) / denominator,
        "max_abs_planned_action": float(diagnostics_totals["max_abs_planned_action"]),
        "max_abs_executed_action": float(diagnostics_totals["max_abs_executed_action"]),
    }


def load_transition_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {key: data[key] for key in data.files}


def save_split_manifest(splits: dict[str, list[str]], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_dataset(dataset_path: str | Path, tasks_path: str | Path, splits_path: str | Path) -> dict[str, Any]:
    dataset = load_transition_dataset(dataset_path)
    tasks = load_tasks(tasks_path)
    splits = load_split_manifest(splits_path)
    return {
        "transition_count": int(dataset["observations"].shape[0]),
        "trajectory_count": int(np.unique(dataset["trajectory_ids"]).shape[0]),
        "task_count": len(tasks),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
    }
