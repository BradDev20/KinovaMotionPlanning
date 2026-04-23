from __future__ import annotations

import argparse
import cProfile
import contextlib
import dataclasses
import io
import json
import multiprocessing as mp
import os
import pstats
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

from src.motion_planning.torch_trajopt import TorchPlannerJob, TorchPlannerResult

import numpy as np

from .collect_dataset_summary import (
    _repair_usage_summary,
    _surrogate_dynamics_checkpoint_summary,
    _surrogate_initial_trajectory_dynamics_summary,
    _surrogate_trajectory_dynamics_summary,
)
from .collect_dataset_types import (
    CollectionJobResult,
    CollectionProgressTracker,
    CollectionTaskDispatch,
    PlannerCoordinatorStop,
)
from .config import EnvConfig
from .dataset import (
    build_transition_dataset,
    deduplicate_records,
    ensure_dir,
    save_metadata,
    save_raw_record,
    save_split_manifest,
    summarize_records,
    trajectory_distance,
)
from .planning import (
    PlannerConfig,
    build_torch_planner_job,
    finalize_planned_trajectory,
    prepare_task_planning_context,
    resolve_collection_device,
    run_torch_planner_batch,
    save_experiment_manifest,
)
from .semantics import artifact_semantics_payload
from .tasks import (
    MIN_SUCCESSFUL_TASKS_PER_FAMILY,
    MIN_SUCCESSFUL_TEST_TASKS_PER_FAMILY,
    MIN_SUCCESSFUL_TRAIN_TASKS_PER_FAMILY,
    MIN_SUCCESSFUL_VAL_TASKS_PER_FAMILY,
    TaskFamilyConfig,
    TaskSampler,
    default_family_mix,
    generate_alpha_values,
    normalize_family_name,
    parse_family_mix,
    regime_families,
    save_tasks,
    split_successful_task_ids_by_family,
)

TASK_PROBE_RESTART_COUNT = 4

MAX_SEEDS_PER_FAMILY = 8
SEED_NOISE_SCALE = 0.03
SEED_DISTINCT_TOL = 0.20


@dataclass(frozen=True)
class SeedEntry:
    trajectory: np.ndarray  # float32, shape=(N, 7)
    start_config: np.ndarray  # float32, shape=(7,)
    goal_config: np.ndarray  # float32, shape=(7,)
    task_id: str = ""
    alpha: float | None = None
    length_cost: float | None = None
    obstacle_cost: float | None = None


def _select_family_seed(
    seeds: list[SeedEntry],
    *,
    start_config: np.ndarray,
    goal_config: np.ndarray,
    rng: np.random.Generator,
    top_k: int = 3,
) -> SeedEntry | None:
    if not seeds:
        return None
    scored: list[tuple[float, int]] = []
    for idx, entry in enumerate(seeds):
        mismatch = float(np.linalg.norm(start_config - entry.start_config) + np.linalg.norm(goal_config - entry.goal_config))
        scored.append((mismatch, idx))
    scored.sort(key=lambda item: item[0])
    k = max(1, min(int(top_k), len(scored)))
    candidates = [idx for _, idx in scored[:k]]
    chosen = int(rng.choice(np.asarray(candidates, dtype=np.int32)))
    return seeds[chosen]


def _select_extreme_seed(
    seeds: list[SeedEntry],
    *,
    kind: str,
) -> SeedEntry | None:
    if not seeds:
        return None
    if kind == "safe":
        candidates = [seed for seed in seeds if seed.obstacle_cost is not None]
        return min(candidates, key=lambda seed: float(seed.obstacle_cost)) if candidates else None
    if kind == "risky":
        candidates = [seed for seed in seeds if seed.length_cost is not None]
        return min(candidates, key=lambda seed: float(seed.length_cost)) if candidates else None
    raise ValueError(f"Unknown extreme seed kind: {kind}")


def _select_diverse_risky_seed(
    seeds: list[SeedEntry],
    *,
    safe_seed: SeedEntry | None,
    top_n: int = 5,
) -> SeedEntry | None:
    candidates = [seed for seed in seeds if seed.length_cost is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda seed: float(seed.length_cost))
    shortlist = candidates[: max(1, min(int(top_n), len(candidates)))]
    if safe_seed is None:
        return shortlist[0]
    # Prefer a risky seed that is topologically distinct from the safe seed when possible.
    return max(
        shortlist,
        key=lambda seed: trajectory_distance(np.asarray(seed.trajectory), np.asarray(safe_seed.trajectory)),
    )


def _select_diverse_risky_record(
    records: list[dict[str, object]],
    *,
    safe_record: dict[str, object] | None,
    top_n: int = 5,
) -> dict[str, object] | None:
    if not records:
        return None
    sorted_by_length = sorted(records, key=lambda rec: (float(rec["length_cost"]), float(rec["obstacle_cost"])))
    shortlist = sorted_by_length[: max(1, min(int(top_n), len(sorted_by_length)))]
    if safe_record is None:
        return shortlist[0]
    safe_traj = np.asarray(safe_record["trajectory"], dtype=np.float32)
    return max(shortlist, key=lambda rec: trajectory_distance(np.asarray(rec["trajectory"], dtype=np.float32), safe_traj))


def _seed_entry_from_record(rec: dict[str, object], *, default_task_id: str) -> SeedEntry:
    traj = np.asarray(rec["trajectory"], dtype=np.float32)
    return SeedEntry(
        trajectory=traj,
        start_config=np.asarray(traj[0], dtype=np.float32),
        goal_config=np.asarray(traj[-1], dtype=np.float32),
        task_id=str(rec.get("task_id", default_task_id)),
        alpha=float(rec.get("alpha")) if rec.get("alpha") is not None else None,
        length_cost=float(rec.get("length_cost")) if rec.get("length_cost") is not None else None,
        obstacle_cost=float(rec.get("obstacle_cost")) if rec.get("obstacle_cost") is not None else None,
    )


def _promote_task_seeds(
    *,
    seed_bank_by_family: dict[str, list[SeedEntry]],
    family: str,
    successful_records: list[dict[str, object]],
    default_task_id: str,
) -> tuple[SeedEntry, SeedEntry] | None:
    if not successful_records:
        return None
    best_safe = min(successful_records, key=lambda rec: (float(rec["obstacle_cost"]), float(rec["length_cost"])))
    best_risky = _select_diverse_risky_record(successful_records, safe_record=best_safe) or best_safe
    safe_seed_entry = _seed_entry_from_record(best_safe, default_task_id=default_task_id)
    risky_seed_entry = _seed_entry_from_record(best_risky, default_task_id=default_task_id)
    _maybe_add_family_seed(seed_bank_by_family, family=str(family), candidate=safe_seed_entry)
    _maybe_add_family_seed(seed_bank_by_family, family=str(family), candidate=risky_seed_entry)
    return safe_seed_entry, risky_seed_entry


def _adapt_seed_trajectory(
    seed: SeedEntry,
    *,
    start_config: np.ndarray,
    goal_config: np.ndarray,
) -> np.ndarray:
    base = np.asarray(seed.trajectory, dtype=np.float32)
    if base.ndim != 2 or base.shape[0] < 2 or base.shape[1] != start_config.shape[0]:
        raise ValueError("Seed trajectory has an unexpected shape.")
    delta_s = start_config.astype(np.float32) - seed.start_config.astype(np.float32)
    delta_g = goal_config.astype(np.float32) - seed.goal_config.astype(np.float32)
    n = int(base.shape[0])
    fractions = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(n, 1)
    shift = (1.0 - fractions) * delta_s.reshape(1, -1) + fractions * delta_g.reshape(1, -1)
    adapted = (base + shift).astype(np.float32, copy=True)
    adapted[0] = start_config.astype(np.float32)
    adapted[-1] = goal_config.astype(np.float32)
    return adapted


def _maybe_add_family_seed(
    seed_bank_by_family: dict[str, list[SeedEntry]],
    *,
    family: str,
    candidate: SeedEntry,
    distinct_tol: float = SEED_DISTINCT_TOL,
    max_seeds: int = MAX_SEEDS_PER_FAMILY,
) -> None:
    seeds = seed_bank_by_family.setdefault(str(family), [])
    # Replace near-duplicates if the candidate is "better" on an extreme metric; otherwise ignore.
    for idx, existing in enumerate(list(seeds)):
        if trajectory_distance(np.asarray(candidate.trajectory), np.asarray(existing.trajectory)) < float(distinct_tol):
            better_safe = (
                candidate.obstacle_cost is not None
                and existing.obstacle_cost is not None
                and float(candidate.obstacle_cost) < float(existing.obstacle_cost) - 1e-6
            )
            better_risky = (
                candidate.length_cost is not None
                and existing.length_cost is not None
                and float(candidate.length_cost) < float(existing.length_cost) - 1e-6
            )
            if better_safe or better_risky:
                seeds[idx] = candidate
            return

    if len(seeds) < int(max_seeds):
        seeds.append(candidate)
        return

    # Bank is full: evict the most redundant non-protected seed (closest to another seed).
    protected: set[int] = set()
    safe_seed = _select_extreme_seed(seeds, kind="safe")
    risky_seed = _select_extreme_seed(seeds, kind="risky")
    if safe_seed is not None:
        for idx, seed in enumerate(seeds):
            if seed is safe_seed:
                protected.add(idx)
                break
    if risky_seed is not None:
        for idx, seed in enumerate(seeds):
            if seed is risky_seed:
                protected.add(idx)
                break

    min_neighbor_distance: list[tuple[float, int]] = []
    for i, seed in enumerate(seeds):
        if i in protected:
            continue
        best = float("inf")
        for j, other in enumerate(seeds):
            if i == j:
                continue
            d = trajectory_distance(np.asarray(seed.trajectory), np.asarray(other.trajectory))
            if d < best:
                best = float(d)
        min_neighbor_distance.append((best, i))

    if not min_neighbor_distance:
        return
    _, evict_idx = min(min_neighbor_distance, key=lambda item: item[0])
    seeds[evict_idx] = candidate


def parse_alpha_grid(alpha_grid: str | None, alpha_count: int, alpha_schedule: str) -> list[float]:
    if alpha_grid:
        return [float(token.strip()) for token in alpha_grid.split(",") if token.strip()]
    return generate_alpha_values(alpha_count=alpha_count, schedule=alpha_schedule)


def parse_seed_values(seed: int, seeds: str | None) -> list[int]:
    if not seeds:
        return [int(seed)]
    return [int(token.strip()) for token in seeds.split(",") if token.strip()]


@contextlib.contextmanager
def _mute_stdio(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _option_provided(flag: str) -> bool:
    return flag in sys.argv[1:]


def resolve_task_families(task_family: str, benchmark_profile: str, geometry_regime: str) -> list[str]:
    normalized = str(task_family).strip().lower()
    if normalized == "all":
        return list(regime_families(benchmark_profile, geometry_regime))
    return [normalize_family_name(normalized)]


def _effective_family_mix(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    parsed = parse_family_mix(args.family_mix)
    return parsed if parsed is not None else default_family_mix(args.benchmark_profile, args.geometry_regime)


def _default_num_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 1)


def apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.benchmark_profile != "max_favoring":
        return
    if not _option_provided("--difficulty"):
        args.difficulty = "hard"
    if not _option_provided("--alpha-count") and not _option_provided("--alpha-grid"):
        args.alpha_count = 15
    if not _option_provided("--alpha-schedule") and not _option_provided("--alpha-grid"):
        args.alpha_schedule = "dense-middle"
    if not _option_provided("--restart-count"):
        args.restart_count = 5
    if not _option_provided("--family-mix"):
        args.family_mix = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect planner-generated trajectories for offline MORL.")
    parser.add_argument("--experiment-name", required=True, help="Experiment name under data/morl/.")
    parser.add_argument("--task-count", type=int, default=30, help="Number of sampled tasks to generate.")
    parser.add_argument("--alpha-grid", type=str, default=None, help="Optional comma-separated alpha values.")
    parser.add_argument("--alpha-count", type=int, default=11, help="Number of alpha values if --alpha-grid is omitted.")
    parser.add_argument("--alpha-schedule", choices=["linear", "dense-middle", "dense-ends"], default="linear", help="Alpha schedule if --alpha-grid is omitted.")
    parser.add_argument("--planner-mode", choices=["sum", "max"], required=True, help="Planner scalarization mode.")
    parser.add_argument("--restart-count", type=int, default=5, help="Number of planner restarts per task/alpha.")
    parser.add_argument("--seed", type=int, default=0, help="Single random seed when --seeds is not provided.")
    parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated collection seeds.")
    parser.add_argument("--output-root", type=str, default="data/morl", help="Root directory for MORL outputs.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional explicit dataset directory. Overrides --output-root/--experiment-name layout.")
    parser.add_argument("--objective-tol", type=float, default=1e-3, help="Objective tolerance for deduplication.")
    parser.add_argument("--path-tol", type=float, default=5e-2, help="Path tolerance for deduplication.")
    parser.add_argument("--route-tol", type=float, default=None, help="Route clustering tolerance. Defaults to 2x path_tol.")
    parser.add_argument("--rho", type=float, default=0.01, help="Tie-breaking rho for weighted max.")
    parser.add_argument("--n-waypoints", type=int, default=25, help="Planner waypoint count.")
    parser.add_argument("--cost-sample-rate", type=int, default=2, help="CPU repair planner cost sampling stride.")
    parser.add_argument("--planner-max-iter", type=int, default=None, help="Compatibility alias for CPU repair iteration cap.")
    parser.add_argument("--planner-max-fun", type=int, default=None, help="Compatibility alias for CPU repair function-eval cap.")
    parser.add_argument("--planner-steps", type=int, default=250, help="Torch optimizer steps per batch.")
    parser.add_argument("--repair-max-iter", type=int, default=None, help="CPU repair iteration cap after surrogate validation fails.")
    parser.add_argument("--repair-max-fun", type=int, default=None, help="CPU repair function-eval cap after surrogate validation fails.")
    parser.add_argument("--device", type=str, default=None, help="Collection planner device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--num-workers", type=int, default=_default_num_workers(), help="Number of CPU task workers.")
    parser.add_argument("--gpu-batch-size", type=int, default=32, help="Maximum micro-batch size for the torch planner.")
    parser.add_argument("--gpu-batch-timeout-ms", type=int, default=10, help="Coordinator micro-batch wait budget in milliseconds.")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-progress collector output.")
    parser.add_argument("--profile", action="store_true", help="Print a cumulative cProfile report for the full collection run.")
    parser.add_argument("--max-steps", type=int, default=25, help="Offline RL environment horizon.")
    parser.add_argument(
        "--task-family",
        choices=[
            "mixed",
            "all",
            "corridor",
            "pinch",
            "stacked",
            "asymmetric",
            "corridor_left_right",
            "pinch_point",
            "stacked_detour",
            "asymmetric_safe_margin",
            "pinch_bottleneck",
            "double_corridor",
            "culdesac_escape",
            "offset_gate",
        ],
        default="mixed",
        help="Task family for the benchmark. Use 'all' to generate one dataset per family in the active regime.",
    )
    parser.add_argument("--family-mix", type=str, default=None, help="Optional explicit family mixture.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium", help="Benchmark difficulty.")
    parser.add_argument("--benchmark-profile", choices=["baseline", "max_favoring"], default="baseline", help="Benchmark profile controlling defaults and planner safety shaping.")
    parser.add_argument("--geometry-regime", choices=["mixed", "convex", "nonconvex"], default="mixed", help="Scene geometry regime.")
    parser.add_argument("--report-size-matched", action="store_true", help="Record that downstream reporting should include size-matched max ablations.")
    args = parser.parse_args()
    apply_profile_defaults(args)
    args.device = resolve_collection_device(
        args.device,
        strict_explicit_cuda=bool(str(args.device).strip().startswith("cuda")) if args.device is not None else False,
    )
    args.num_workers = max(1, int(args.num_workers))
    args.gpu_batch_size = max(1, int(args.gpu_batch_size))
    args.gpu_batch_timeout_ms = max(1, int(args.gpu_batch_timeout_ms))
    args.planner_steps = max(1, int(args.planner_steps))
    args.repair_max_iter = int(args.repair_max_iter) if args.repair_max_iter is not None else int(args.planner_max_iter or 40)
    args.repair_max_fun = int(args.repair_max_fun) if args.repair_max_fun is not None else int(args.planner_max_fun or 160)
    return args


def _print_cumulative_profile(profiler: cProfile.Profile) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("tottime").print_stats(15)
    print("\nCollection profile (top 15 by total time):")
    print(stream.getvalue().rstrip())


def _seed_output_dir(mode_root: Path, seed_value: int, multi_seed: bool) -> Path:
    return ensure_dir(mode_root / f"seed_{seed_value:04d}") if multi_seed else ensure_dir(mode_root)


class CollectionSupportError(RuntimeError):
    def __init__(self, message: str, *, dataset_dir: Path, support_check: dict[str, object]):
        super().__init__(message)
        self.dataset_dir = Path(dataset_dir)
        self.support_check = support_check


def _expected_families(task_family: str, family_mix: tuple[tuple[str, float], ...]) -> list[str]:
    normalized = normalize_family_name(task_family)
    if normalized in {"mixed", "all"}:
        return [name for name, _ in family_mix]
    return [normalized]


def _support_summary_for_records(
    records: list[dict[str, object]],
    *,
    expected_families: list[str],
    seed_value: int,
) -> dict[str, object]:
    successful_task_ids_by_family: dict[str, list[str]] = {family: [] for family in expected_families}
    for record in records:
        task_spec = record["task_spec"]
        family = str(task_spec["family"])
        successful_task_ids_by_family.setdefault(family, [])
        successful_task_ids_by_family[family].append(str(task_spec["task_id"]))
    successful_task_ids_by_family = {
        family: sorted(set(task_ids))
        for family, task_ids in sorted(successful_task_ids_by_family.items())
    }
    successful_task_counts_by_family = {
        family: len(task_ids)
        for family, task_ids in successful_task_ids_by_family.items()
    }
    support_check: dict[str, object] = {
        "passed": False,
        "failure_reason": None,
        "expected_families": list(expected_families),
        "minimum_successful_tasks_per_family": MIN_SUCCESSFUL_TASKS_PER_FAMILY,
        "minimum_successful_train_tasks_per_family": MIN_SUCCESSFUL_TRAIN_TASKS_PER_FAMILY,
        "minimum_successful_val_tasks_per_family": MIN_SUCCESSFUL_VAL_TASKS_PER_FAMILY,
        "minimum_successful_test_tasks_per_family": MIN_SUCCESSFUL_TEST_TASKS_PER_FAMILY,
        "successful_task_ids_by_family": successful_task_ids_by_family,
        "successful_task_counts_by_family": successful_task_counts_by_family,
        "split_task_ids_by_family": {"train": {}, "val": {}, "test": {}},
    }
    try:
        split_payload = split_successful_task_ids_by_family(
            successful_task_ids_by_family,
            seed=seed_value,
            min_train_tasks=MIN_SUCCESSFUL_TRAIN_TASKS_PER_FAMILY,
            min_val_tasks=MIN_SUCCESSFUL_VAL_TASKS_PER_FAMILY,
            min_test_tasks=MIN_SUCCESSFUL_TEST_TASKS_PER_FAMILY,
        )
    except ValueError as exc:
        support_check["failure_reason"] = str(exc)
        return support_check

    support_check["passed"] = True
    support_check["splits"] = split_payload["splits"]
    support_check["split_task_ids_by_family"] = split_payload["split_by_family"]
    return support_check


def _planner_config_from_args(args: argparse.Namespace) -> PlannerConfig:
    safety_aggregate = "max" if args.benchmark_profile == "max_favoring" else "avg"
    safety_decay_rate = 50.0 if args.benchmark_profile == "max_favoring" else 40.0
    safety_bias = 0.05 if args.benchmark_profile == "max_favoring" else 0.02
    safety_collision_penalty = 3.0 if args.benchmark_profile == "max_favoring" else 1.0
    return PlannerConfig(
        n_waypoints=args.n_waypoints,
        cost_sample_rate=args.cost_sample_rate,
        rho=args.rho,
        planner_max_iter=args.planner_max_iter,
        planner_max_fun=args.planner_max_fun,
        benchmark_profile=args.benchmark_profile,
        geometry_regime=args.geometry_regime,
        safety_aggregate=safety_aggregate,
        safety_decay_rate=safety_decay_rate,
        safety_bias=safety_bias,
        safety_collision_penalty=safety_collision_penalty,
        device=args.device,
        gpu_batch_size=args.gpu_batch_size,
        gpu_batch_timeout_ms=args.gpu_batch_timeout_ms,
        planner_steps=args.planner_steps,
        repair_max_iter=args.repair_max_iter,
        repair_max_fun=args.repair_max_fun,
    )


def _failure_payload(task, alpha: float, restart_index: int, error: Exception) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "alpha": float(alpha),
        "restart_index": int(restart_index),
        "family": task.family,
        "geometry_regime": task.geometry_regime,
        "benchmark_profile": task.benchmark_profile,
        "error": str(error),
    }


def _skipped_failure_payload(
    task,
    *,
    alpha: float,
    restart_index: int,
    attempted_jobs: int,
    total_jobs: int,
) -> dict[str, object]:
    payload = _failure_payload(
        task,
        alpha,
        restart_index,
        RuntimeError(
            f"Skipped remaining restarts after {attempted_jobs}/{total_jobs} probe jobs produced no valid trajectory for this task."
        ),
    )
    payload["probe_skipped"] = True
    payload["probe_attempted_jobs"] = int(attempted_jobs)
    payload["probe_total_jobs"] = int(total_jobs)
    return payload


def _chunked(items: list[tuple[float, int, TorchPlannerJob]], size: int):
    chunk_size = max(int(size), 1)
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _build_task_jobs(
    dispatch: CollectionTaskDispatch,
    *,
    context,
    planner_config: PlannerConfig,
    worker_id: int,
) -> list[tuple[float, int, TorchPlannerJob]]:
    jobs: list[tuple[float, int, TorchPlannerJob]] = []
    for restart_index in range(dispatch.restart_count):
        for alpha_index, alpha in enumerate(dispatch.alpha_values):
            order_index = dispatch.order_offset + alpha_index * dispatch.restart_count + restart_index
            jobs.append(
                (
                    float(alpha),
                    int(restart_index),
                    build_torch_planner_job(
                        context,
                        alpha=float(alpha),
                        mode=dispatch.mode,
                        restart_index=restart_index,
                        order_index=order_index,
                        worker_id=worker_id,
                        planner_config=planner_config,
                    ),
                )
            )
    return jobs


def _probe_job_count(dispatch: CollectionTaskDispatch) -> int:
    return len(dispatch.alpha_values) * min(int(dispatch.restart_count), TASK_PROBE_RESTART_COUNT)


def _task_has_success(results: list[CollectionJobResult]) -> bool:
    return any(result.record is not None for result in results)


def _skip_remaining_task_jobs(
    dispatch: CollectionTaskDispatch,
    *,
    job_specs: list[tuple[float, int, TorchPlannerJob]],
    attempted_jobs: int,
) -> list[CollectionJobResult]:
    total_jobs = len(job_specs)
    skipped_specs = job_specs[attempted_jobs:]
    return [
        CollectionJobResult(
            order_index=job.order_index,
            task_index=dispatch.task_index,
            failure=_skipped_failure_payload(
                dispatch.task,
                alpha=alpha,
                restart_index=restart_index,
                attempted_jobs=attempted_jobs,
                total_jobs=total_jobs,
            ),
        )
        for alpha, restart_index, job in skipped_specs
    ]


def _finalize_job_results(
    *,
    dispatch: CollectionTaskDispatch,
    context,
    job_specs: list[tuple[float, int, TorchPlannerJob]],
    responses: dict[str, TorchPlannerResult],
    planner_config: PlannerConfig,
) -> list[CollectionJobResult]:
    results: list[CollectionJobResult] = []
    for alpha, restart_index, job in job_specs:
        planner_result = responses[job.request_id]
        try:
            record = finalize_planned_trajectory(
                context,
                planner_result=planner_result,
                alpha=alpha,
                mode=dispatch.mode,
                restart_index=restart_index,
                planner_config=planner_config,
            )
            results.append(CollectionJobResult(order_index=job.order_index, task_index=dispatch.task_index, record=record))
        except Exception as exc:  # pragma: no cover - runtime planner failures depend on MuJoCo optimization
            results.append(
                CollectionJobResult(
                    order_index=job.order_index,
                    task_index=dispatch.task_index,
                    failure=_failure_payload(context.task, alpha, restart_index, exc),
                )
            )
    return results


def _collect_task_sequential(
    dispatch: CollectionTaskDispatch,
    *,
    scene_dir: Path,
    planner_config: PlannerConfig,
    seed_bank_by_family: dict[str, list[SeedEntry]] | None = None,
) -> list[CollectionJobResult]:
    try:
        context = prepare_task_planning_context(dispatch.task, scene_dir=scene_dir, planner_config=planner_config)
    except Exception as exc:
        return [
            CollectionJobResult(
                order_index=dispatch.order_offset + alpha_index * dispatch.restart_count + restart_index,
                task_index=dispatch.task_index,
                failure=_failure_payload(dispatch.task, alpha, restart_index, exc),
            )
            for alpha_index, alpha in enumerate(dispatch.alpha_values)
            for restart_index in range(dispatch.restart_count)
        ]

    job_specs = _build_task_jobs(dispatch, context=context, planner_config=planner_config, worker_id=-1)
    probe_count = _probe_job_count(dispatch)
    probe_job_specs = job_specs[:probe_count]
    remaining_job_specs = job_specs[probe_count:]

    # Optionally apply per-family legal seed warm-starts to the probe jobs.
    # Keep restart_index=0 unseeded to preserve RRT warm-start exploration.
    if seed_bank_by_family is not None:
        family = str(dispatch.task.family)
        seeds = seed_bank_by_family.get(family, [])
        if seeds:
            rng = np.random.default_rng(int(dispatch.task.planner_seed))
            start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
            goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
            safe_seed = _select_extreme_seed(seeds, kind="safe")
            risky_seed = _select_diverse_risky_seed(seeds, safe_seed=safe_seed)

            updated_probe: list[tuple[float, int, TorchPlannerJob]] = []
            for alpha, restart_index, job in probe_job_specs:
                if int(restart_index) == 0:
                    updated_probe.append((alpha, restart_index, job))
                    continue
                style = "random"
                seed = None
                if int(restart_index) == 1:
                    style = "safe"
                    seed = safe_seed
                elif int(restart_index) == 2:
                    style = "risky"
                    seed = risky_seed
                if seed is None:
                    seed = _select_family_seed(seeds, start_config=start_cfg, goal_config=goal_cfg, rng=rng)
                if seed is None:
                    updated_probe.append((alpha, restart_index, job))
                    continue
                try:
                    warm = _adapt_seed_trajectory(seed, start_config=start_cfg, goal_config=goal_cfg)
                    updated_probe.append(
                        (
                            alpha,
                            restart_index,
                            dataclasses.replace(
                                job,
                                warm_start_trajectory=warm,
                                warm_start_noise_scale=float(SEED_NOISE_SCALE),
                                warm_start_tag=f"family_seed_{style}",
                            ),
                        )
                    )
                except Exception:
                    updated_probe.append((alpha, restart_index, job))
            probe_job_specs = updated_probe
    responses: dict[str, TorchPlannerResult] = {}
    for batch in _chunked(probe_job_specs, planner_config.gpu_batch_size):
        batch_results = run_torch_planner_batch([job for _, _, job in batch], planner_config=planner_config)
        responses.update({result.request_id: result for result in batch_results})
    finalized_probe = _finalize_job_results(
        dispatch=dispatch,
        context=context,
        job_specs=probe_job_specs,
        responses=responses,
        planner_config=planner_config,
    )

    task_seeds: tuple[SeedEntry, SeedEntry] | None = None
    if seed_bank_by_family is not None and _task_has_success(finalized_probe):
        successful_records = [item.record for item in finalized_probe if item.record is not None]
        task_seeds = _promote_task_seeds(
            seed_bank_by_family=seed_bank_by_family,
            family=str(dispatch.task.family),
            successful_records=successful_records,
            default_task_id=str(dispatch.task.task_id),
        )
    if remaining_job_specs and not _task_has_success(finalized_probe):
        return finalized_probe + _skip_remaining_task_jobs(
            dispatch,
            job_specs=job_specs,
            attempted_jobs=probe_count,
        )

    if seed_bank_by_family is not None and remaining_job_specs and task_seeds is not None:
        safe_seed_entry, risky_seed_entry = task_seeds
        start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
        goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
        updated_remaining: list[tuple[float, int, TorchPlannerJob]] = []
        for alpha, restart_index, job in remaining_job_specs:
            chosen = safe_seed_entry if float(alpha) <= 0.5 else risky_seed_entry
            style = "safe" if chosen is safe_seed_entry else "risky"
            try:
                warm = _adapt_seed_trajectory(chosen, start_config=start_cfg, goal_config=goal_cfg)
                updated_remaining.append(
                    (
                        alpha,
                        restart_index,
                        dataclasses.replace(
                            job,
                            warm_start_trajectory=warm,
                            warm_start_noise_scale=float(SEED_NOISE_SCALE),
                            warm_start_tag=f"task_seed_{style}",
                        ),
                    )
                )
            except Exception:
                updated_remaining.append((alpha, restart_index, job))
        remaining_job_specs = updated_remaining

    for batch in _chunked(remaining_job_specs, planner_config.gpu_batch_size):
        batch_results = run_torch_planner_batch([job for _, _, job in batch], planner_config=planner_config)
        responses.update({result.request_id: result for result in batch_results})
    return finalized_probe + _finalize_job_results(
        dispatch=dispatch,
        context=context,
        job_specs=remaining_job_specs,
        responses=responses,
        planner_config=planner_config,
    )


def _planner_coordinator_main(
    request_queue,
    response_queues,
    worker_count: int,
    planner_config: PlannerConfig,
    quiet: bool = False,
) -> None:
    with _mute_stdio(bool(quiet)):
        stop_count = 0
        while stop_count < worker_count:
            try:
                first_item = request_queue.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(first_item, PlannerCoordinatorStop):
                stop_count += 1
                continue

            batch: list[TorchPlannerJob] = [first_item]
            deadline = time.perf_counter() + (float(planner_config.gpu_batch_timeout_ms) / 1000.0)
            while len(batch) < int(planner_config.gpu_batch_size):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    queued_item = request_queue.get(timeout=remaining)
                except Empty:
                    break
                if isinstance(queued_item, PlannerCoordinatorStop):
                    stop_count += 1
                    continue
                batch.append(queued_item)

            for result in run_torch_planner_batch(batch, planner_config=planner_config):
                response_queues[result.worker_id].put(result)


def _collection_worker_main(
    worker_id: int,
    task_queue,
    planner_request_queue,
    planner_response_queue,
    result_queue,
    scene_dir: str,
    planner_config: PlannerConfig,
    quiet: bool = False,
) -> None:
    with _mute_stdio(bool(quiet)):
        seed_bank_by_family: dict[str, list[SeedEntry]] = {}
        while True:
            dispatch = task_queue.get()
            if dispatch is None:
                planner_request_queue.put(PlannerCoordinatorStop(worker_id=worker_id))
                return

            try:
                context = prepare_task_planning_context(dispatch.task, scene_dir=scene_dir, planner_config=planner_config)
            except Exception as exc:
                for alpha_index, alpha in enumerate(dispatch.alpha_values):
                    for restart_index in range(dispatch.restart_count):
                        result_queue.put(
                            CollectionJobResult(
                                order_index=dispatch.order_offset + alpha_index * dispatch.restart_count + restart_index,
                                task_index=dispatch.task_index,
                                failure=_failure_payload(dispatch.task, alpha, restart_index, exc),
                            )
                        )
                continue

            job_specs = _build_task_jobs(dispatch, context=context, planner_config=planner_config, worker_id=worker_id)
            probe_count = _probe_job_count(dispatch)
            probe_job_specs = job_specs[:probe_count]
            remaining_job_specs = job_specs[probe_count:]

            # Optional per-family seed warm start for probe jobs.
            family = str(dispatch.task.family)
            seeds = seed_bank_by_family.get(family, [])
            if seeds:
                rng = np.random.default_rng(int(dispatch.task.planner_seed))
                start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
                goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
                safe_seed = _select_extreme_seed(seeds, kind="safe")
                risky_seed = _select_diverse_risky_seed(seeds, safe_seed=safe_seed)
                updated_probe: list[tuple[float, int, TorchPlannerJob]] = []
                for alpha, restart_index, job in probe_job_specs:
                    if int(restart_index) == 0:
                        updated_probe.append((alpha, restart_index, job))
                        continue
                    style = "random"
                    seed = None
                    if int(restart_index) == 1:
                        style = "safe"
                        seed = safe_seed
                    elif int(restart_index) == 2:
                        style = "risky"
                        seed = risky_seed
                    if seed is None:
                        seed = _select_family_seed(seeds, start_config=start_cfg, goal_config=goal_cfg, rng=rng)
                    if seed is None:
                        updated_probe.append((alpha, restart_index, job))
                        continue
                    try:
                        warm = _adapt_seed_trajectory(seed, start_config=start_cfg, goal_config=goal_cfg)
                        updated_probe.append(
                            (
                                alpha,
                                restart_index,
                                dataclasses.replace(
                                    job,
                                    warm_start_trajectory=warm,
                                    warm_start_noise_scale=float(SEED_NOISE_SCALE),
                                    warm_start_tag=f"family_seed_{style}",
                                ),
                            )
                        )
                    except Exception:
                        updated_probe.append((alpha, restart_index, job))
                probe_job_specs = updated_probe
            for _, _, job in probe_job_specs:
                planner_request_queue.put(job)

            responses: dict[str, TorchPlannerResult] = {}
            while len(responses) < len(probe_job_specs):
                planner_result = planner_response_queue.get()
                responses[planner_result.request_id] = planner_result

            finalized_probe = _finalize_job_results(
                dispatch=dispatch,
                context=context,
                job_specs=probe_job_specs,
                responses=responses,
                planner_config=planner_config,
            )
            for finalized in finalized_probe:
                result_queue.put(finalized)

            task_seeds: tuple[SeedEntry, SeedEntry] | None = None
            if _task_has_success(finalized_probe):
                successful_records = [item.record for item in finalized_probe if item.record is not None]
                task_seeds = _promote_task_seeds(
                    seed_bank_by_family=seed_bank_by_family,
                    family=family,
                    successful_records=successful_records,
                    default_task_id=str(dispatch.task.task_id),
                )
            if remaining_job_specs and not _task_has_success(finalized_probe):
                for skipped in _skip_remaining_task_jobs(
                    dispatch,
                    job_specs=job_specs,
                    attempted_jobs=probe_count,
                ):
                    result_queue.put(skipped)
                continue

            # Task-local seed for remaining jobs + promote into the family bank.
            if remaining_job_specs and task_seeds is not None:
                safe_seed_entry, risky_seed_entry = task_seeds
                start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
                goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
                updated_remaining: list[tuple[float, int, TorchPlannerJob]] = []
                for alpha, restart_index, job in remaining_job_specs:
                    chosen = safe_seed_entry if float(alpha) <= 0.5 else risky_seed_entry
                    style = "safe" if chosen is safe_seed_entry else "risky"
                    try:
                        warm = _adapt_seed_trajectory(chosen, start_config=start_cfg, goal_config=goal_cfg)
                        updated_remaining.append(
                            (
                                alpha,
                                restart_index,
                                dataclasses.replace(
                                    job,
                                    warm_start_trajectory=warm,
                                    warm_start_noise_scale=float(SEED_NOISE_SCALE),
                                    warm_start_tag=f"task_seed_{style}",
                                ),
                            )
                        )
                    except Exception:
                        updated_remaining.append((alpha, restart_index, job))
                remaining_job_specs = updated_remaining

            for _, _, job in remaining_job_specs:
                planner_request_queue.put(job)
            while len(responses) < len(job_specs):
                planner_result = planner_response_queue.get()
                responses[planner_result.request_id] = planner_result

            for finalized in _finalize_job_results(
                dispatch=dispatch,
                context=context,
                job_specs=remaining_job_specs,
                responses=responses,
                planner_config=planner_config,
            ):
                result_queue.put(finalized)


def _collect_task_results_parallel(
    tasks: list,
    *,
    alpha_values: list[float],
    restart_count: int,
    scene_dir: Path,
    planner_mode: str,
    planner_config: PlannerConfig,
    num_workers: int,
    progress_tracker: CollectionProgressTracker | None = None,
    quiet: bool = False,
) -> list[CollectionJobResult]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    planner_request_queue = ctx.Queue()
    result_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(num_workers)]
    jobs_per_task = len(alpha_values) * restart_count
    coordinator = None
    workers: list[mp.Process] = []

    def _close_queue(queue) -> None:
        try:
            queue.close()
        except Exception:
            pass
        try:
            queue.cancel_join_thread()
        except Exception:
            pass

    def _shutdown_processes() -> None:
        active_processes = [process for process in [coordinator, *workers] if process is not None]
        for process in active_processes:
            if process.is_alive():
                process.terminate()
        for process in active_processes:
            try:
                process.join(timeout=5.0)
            except Exception:
                pass
        for process in active_processes:
            if process.is_alive() and hasattr(process, "kill"):
                try:
                    process.kill()
                except Exception:
                    pass
        for process in active_processes:
            try:
                process.join(timeout=1.0)
            except Exception:
                pass

    try:
        coordinator = ctx.Process(
            target=_planner_coordinator_main,
            args=(planner_request_queue, response_queues, num_workers, planner_config, bool(quiet)),
        )
        workers = [
            ctx.Process(
                target=_collection_worker_main,
                args=(
                    worker_id,
                    task_queue,
                    planner_request_queue,
                    response_queues[worker_id],
                    result_queue,
                    str(scene_dir),
                    planner_config,
                    bool(quiet),
                ),
            )
            for worker_id in range(num_workers)
        ]
        coordinator.start()
        for worker in workers:
            worker.start()

        for task_index, task in enumerate(tasks):
            task_queue.put(
                CollectionTaskDispatch(
                    task=task,
                    task_index=task_index,
                    alpha_values=tuple(alpha_values),
                    restart_count=int(restart_count),
                    mode=planner_mode,
                    order_offset=task_index * jobs_per_task,
                )
            )
        for _ in workers:
            task_queue.put(None)

        total_job_count = len(tasks) * jobs_per_task
        collected_results = []
        for _ in range(total_job_count):
            result = result_queue.get()
            collected_results.append(result)
            if progress_tracker is not None:
                progress_tracker.advance(result)

        for worker in workers:
            worker.join()
        coordinator.join()
        return collected_results
    except BaseException:
        _shutdown_processes()
        raise
    finally:
        _close_queue(task_queue)
        _close_queue(planner_request_queue)
        _close_queue(result_queue)
        for queue in response_queues:
            _close_queue(queue)


def collect_one_seed(
    args: argparse.Namespace,
    seed_value: int,
    mode_root: Path,
    alpha_values: list[float],
    multi_seed: bool,
    task_family: str,
    family_mix: tuple[tuple[str, float], ...],
) -> dict[str, object]:
    experiment_dir = _seed_output_dir(mode_root, seed_value=seed_value, multi_seed=multi_seed)
    raw_dir = ensure_dir(experiment_dir / "raw")
    scene_dir = ensure_dir(experiment_dir / "scenes")

    family_config = TaskFamilyConfig(
        task_family=task_family,
        difficulty=args.difficulty,
        benchmark_profile=args.benchmark_profile,
        geometry_regime=args.geometry_regime,
        family_mix=family_mix,
    )
    task_sampler = TaskSampler(seed=seed_value, family_config=family_config)
    tasks = task_sampler.sample_tasks(args.task_count)
    progress_tracker = CollectionProgressTracker(tasks=tasks, alpha_values=alpha_values, restart_count=int(args.restart_count))
    progress_tracker.start()
    save_tasks(tasks, experiment_dir / "tasks.json")
    splits_path = experiment_dir / "splits.json"
    expected_families = _expected_families(task_family, family_mix)
    planner_config = _planner_config_from_args(args)

    jobs_per_task = len(alpha_values) * args.restart_count
    try:
        with _mute_stdio(bool(args.quiet)):
            if args.num_workers <= 1:
                ordered_results = []
                seed_bank_by_family: dict[str, list[SeedEntry]] = {}
                for task_index, task in enumerate(tasks):
                    dispatch = CollectionTaskDispatch(
                        task=task,
                        task_index=task_index,
                        alpha_values=tuple(alpha_values),
                        restart_count=int(args.restart_count),
                        mode=args.planner_mode,
                        order_offset=task_index * jobs_per_task,
                    )
                    task_results = _collect_task_sequential(
                        dispatch,
                        scene_dir=scene_dir,
                        planner_config=planner_config,
                        seed_bank_by_family=seed_bank_by_family,
                    )
                    ordered_results.extend(task_results)
                    for result in task_results:
                        progress_tracker.advance(result)
            else:
                ordered_results = _collect_task_results_parallel(
                    tasks,
                    alpha_values=alpha_values,
                    restart_count=int(args.restart_count),
                    scene_dir=scene_dir,
                    planner_mode=args.planner_mode,
                    planner_config=planner_config,
                    num_workers=int(args.num_workers),
                    progress_tracker=progress_tracker,
                    quiet=bool(args.quiet),
                )
    finally:
        progress_tracker.finish()

    ordered_results.sort(key=lambda item: item.order_index)
    records = [item.record for item in ordered_results if item.record is not None]
    failed = [item.failure for item in ordered_results if item.failure is not None]

    for record in records:
        save_raw_record(record, raw_dir)

    deduped = deduplicate_records(records, objective_tol=args.objective_tol, path_tol=args.path_tol)
    save_metadata(deduped, experiment_dir / "trajectory_metadata.json")
    transition_summary = build_transition_dataset(
        deduped,
        scene_dir=scene_dir,
        output_path=experiment_dir / "transitions.npz",
        env_config=EnvConfig(max_steps=args.max_steps),
    )
    if transition_summary["transition_count"] == 0 and not bool(args.quiet):
        print(
            f"[seed={seed_value}] No usable transitions were generated for family={task_family}. "
            f"raw={len(records)}, deduplicated={len(deduped)}, failed={len(failed)}"
        )
    record_summary = summarize_records(
        deduped,
        objective_tol=args.objective_tol,
        path_tol=args.path_tol,
        route_tol=args.route_tol,
    )
    successful_task_ids = sorted({str(record["task_spec"]["task_id"]) for record in deduped})
    successful_task_count = len(successful_task_ids)
    zero_success_task_count = max(len(tasks) - successful_task_count, 0)
    probe_skipped_failures = [item for item in failed if bool(item.get("probe_skipped"))]
    probe_skipped_task_count = len({str(item["task_id"]) for item in probe_skipped_failures})
    repair_summary = _repair_usage_summary(records)
    surrogate_initial_dynamics_summary = _surrogate_initial_trajectory_dynamics_summary(records)
    surrogate_dynamics_summary = _surrogate_trajectory_dynamics_summary(records)
    surrogate_checkpoint_dynamics_summary = _surrogate_dynamics_checkpoint_summary(records)
    support_check = _support_summary_for_records(
        deduped,
        expected_families=expected_families,
        seed_value=seed_value,
    )
    if bool(support_check["passed"]):
        save_split_manifest(support_check["splits"], splits_path)
    elif splits_path.exists():
        splits_path.unlink()

    summary = {
        **transition_summary,
        "raw_trajectory_count": len(records),
        "deduplicated_trajectory_count": len(deduped),
        "failed_count": len(failed),
        "requested_task_count": len(tasks),
        "successful_task_count": successful_task_count,
        "zero_success_task_count": zero_success_task_count,
        "successful_task_ids": successful_task_ids,
        "probe_skipped_task_count": probe_skipped_task_count,
        "probe_skipped_job_count": len(probe_skipped_failures),
        **repair_summary,
        **surrogate_initial_dynamics_summary,
        **surrogate_dynamics_summary,
        **surrogate_checkpoint_dynamics_summary,
        "planner_mode": args.planner_mode,
        "planner_backend": "torch",
        "planner_device": planner_config.device,
        "num_workers": int(args.num_workers),
        "alpha_grid": alpha_values,
        "task_family": task_family,
        "difficulty": args.difficulty,
        "benchmark_profile": args.benchmark_profile,
        "geometry_regime": args.geometry_regime,
        "safety_aggregate": planner_config.safety_aggregate,
        "family_mix": [[name, weight] for name, weight in family_mix],
        "seed": seed_value,
        "support_check": support_check,
        **artifact_semantics_payload(),
        **record_summary,
    }
    (experiment_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "experiment_name": args.experiment_name,
        "planner_mode": args.planner_mode,
        "planner_backend": "torch",
        "task_count": args.task_count,
        "alpha_grid": alpha_values,
        "restart_count": args.restart_count,
        "seed": seed_value,
        "planner_config": planner_config.__dict__,
        "env_config": {"max_steps": args.max_steps},
        "family_config": {
            "task_family": task_family,
            "family_mix": list(family_mix),
            "difficulty": args.difficulty,
            "benchmark_profile": args.benchmark_profile,
            "geometry_regime": args.geometry_regime,
        },
        "report_size_matched": bool(args.report_size_matched),
        **artifact_semantics_payload(),
        "summary": summary,
        "support_check": support_check,
        "failed": failed,
    }
    save_experiment_manifest(manifest, experiment_dir / "experiment_config.json")
    if not bool(support_check["passed"]):
        raise CollectionSupportError(
            str(support_check["failure_reason"]),
            dataset_dir=experiment_dir,
            support_check=support_check,
        )
    if not bool(args.quiet):
        print(
            f"Saved dataset to {experiment_dir} "
            f"(raw={len(records)}, deduplicated={len(deduped)}, failed={len(failed)})"
        )
    return {
        "seed": seed_value,
        "task_family": task_family,
        "path": str(experiment_dir),
        "summary": summary,
        "support_check": support_check,
    }


def _run_collection(args: argparse.Namespace):
    alpha_values = parse_alpha_grid(args.alpha_grid, args.alpha_count, args.alpha_schedule)
    mode_root = ensure_dir(Path(args.dataset_dir)) if args.dataset_dir else ensure_dir(Path(args.output_root) / args.experiment_name / args.planner_mode)
    seed_values = parse_seed_values(args.seed, args.seeds)
    multi_seed = len(seed_values) > 1 or bool(args.seeds)
    task_families = resolve_task_families(args.task_family, args.benchmark_profile, args.geometry_regime)
    family_mix = _effective_family_mix(args)

    if len(task_families) == 1:
        task_family = task_families[0]
        active_mix = family_mix if task_family in {"mixed", "all"} else ((task_family, 1.0),)
        seed_runs = [
            collect_one_seed(
                args,
                seed_value,
                mode_root=mode_root,
                alpha_values=alpha_values,
                multi_seed=multi_seed,
                task_family=task_family,
                family_mix=active_mix,
            )
            for seed_value in seed_values
        ]
        aggregate_summary = {
            "experiment_name": args.experiment_name,
            "planner_mode": args.planner_mode,
            "planner_backend": "torch",
            "planner_device": args.device,
            "seed_count": len(seed_runs),
            "alpha_grid": alpha_values,
            "task_family": task_family,
            "difficulty": args.difficulty,
            "benchmark_profile": args.benchmark_profile,
            "geometry_regime": args.geometry_regime,
            "safety_aggregate": "max" if args.benchmark_profile == "max_favoring" else "avg",
            "family_mix": [[name, weight] for name, weight in active_mix],
            "seed_breakdown": {f"seed_{item['seed']:04d}": item["summary"] for item in seed_runs},
            "report_size_matched": bool(args.report_size_matched),
            **artifact_semantics_payload(),
        }
    else:
        family_runs: dict[str, dict[str, object]] = {}
        for task_family in task_families:
            family_root = ensure_dir(mode_root / task_family)
            seed_runs = [
                collect_one_seed(
                    args,
                    seed_value,
                    mode_root=family_root,
                    alpha_values=alpha_values,
                    multi_seed=multi_seed,
                    task_family=task_family,
                    family_mix=((task_family, 1.0),),
                )
                for seed_value in seed_values
            ]
            family_runs[task_family] = {
                "seed_count": len(seed_runs),
                "path": str(family_root),
                "seed_breakdown": {f"seed_{item['seed']:04d}": item["summary"] for item in seed_runs},
            }
        aggregate_summary = {
            "experiment_name": args.experiment_name,
            "planner_mode": args.planner_mode,
            "planner_backend": "torch",
            "planner_device": args.device,
            "seed_count": len(seed_values),
            "alpha_grid": alpha_values,
            "task_family": "all",
            "difficulty": args.difficulty,
            "benchmark_profile": args.benchmark_profile,
            "geometry_regime": args.geometry_regime,
            "safety_aggregate": "max" if args.benchmark_profile == "max_favoring" else "avg",
            "family_mix": [[name, weight] for name, weight in family_mix],
            "family_breakdown": family_runs,
            "report_size_matched": bool(args.report_size_matched),
            **artifact_semantics_payload(),
        }

    aggregate_summary_path = mode_root / "aggregate_summary.json"
    aggregate_summary_path.write_text(json.dumps(aggregate_summary, indent=2), encoding="utf-8")
    return {
        "mode_root": str(mode_root),
        "aggregate_summary_path": str(aggregate_summary_path),
        "aggregate_summary": aggregate_summary,
    }


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], *argv]
    try:
        args = parse_args()
    finally:
        if argv is not None:
            sys.argv = original_argv

    if not bool(args.profile):
        return _run_collection(args)

    profiler = cProfile.Profile()
    try:
        return profiler.runcall(_run_collection, args)
    finally:
        _print_cumulative_profile(profiler)


if __name__ == "__main__":
    main()
