"""
Main script for collecting trajectory datasets for offline MORL.
This orchestrates the whole process: sampling tasks, running planners 
(with a seed bank for speed), and saving the results into a dataset.
"""
from __future__ import annotations

import argparse
import cProfile
import contextlib
import io
import json
import os
import pstats
import sys
from pathlib import Path

import numpy as np

from .collection.seed_bank import load_seed_bank, _maybe_add_family_seed, _seed_entry_from_record, save_seed_bank, SeedEntry
from .collection.summary import (
    _repair_usage_summary,
    _surrogate_dynamics_checkpoint_summary,
    _surrogate_initial_trajectory_dynamics_summary,
    _surrogate_trajectory_dynamics_summary,
)
from .collection.types import (
    CollectionJobResult,
    CollectionProgressTracker,
    CollectionTaskDispatch,
)
from .collection.workers import (
    _collect_task_results_parallel,
    _collect_task_sequential,
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
)
from .planning import (
    PlannerConfig,
    resolve_collection_device,
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

def parse_alpha_grid(alpha_grid: str | None, alpha_count: int, alpha_schedule: str) -> list[float]:
    """Turn a string of alphas or a schedule into a list of floats."""
    if alpha_grid:
        return [float(token.strip()) for token in alpha_grid.split(",") if token.strip()]
    return generate_alpha_values(alpha_count=alpha_count, schedule=alpha_schedule)

def parse_seed_values(seed: int, seeds: str | None) -> list[int]:
    """Helper to parse one or many seeds from the CLI args."""
    if not seeds:
        return [int(seed)]
    return [int(token.strip()) for token in seeds.split(",") if token.strip()]

@contextlib.contextmanager
def _mute_stdio(enabled: bool):
    """Context manager to shut up the standard output/error if requested."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def _option_provided(flag: str) -> bool:
    """Checks if a specific flag was passed in sys.argv."""
    return flag in sys.argv[1:]

def resolve_task_families(task_family: str, benchmark_profile: str, geometry_regime: str) -> list[str]:
    """Figure out which families we're actually running."""
    normalized = str(task_family).strip().lower()
    if normalized == "all":
        return list(regime_families(benchmark_profile, geometry_regime))
    return [normalize_family_name(normalized)]

def _effective_family_mix(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    """Gets the mix of families to sample from."""
    parsed = parse_family_mix(args.family_mix)
    return parsed if parsed is not None else default_family_mix(args.benchmark_profile, args.geometry_regime)

def _default_num_workers() -> int:
    """Guess a good number of workers based on CPU count."""
    return max(1, (os.cpu_count() or 1) - 1)

def apply_profile_defaults(args: argparse.Namespace) -> None:
    """Apply some 'smart' defaults based on the chosen benchmark profile."""
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
    """The big argument parser for the collection script."""
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
    parser.add_argument(
        "--seed-bank-dir",
        type=Path,
        default=None,
        help="Global seed bank directory shared across experiments. "
             "If omitted, uses {mode_root}/seed_bank/ (per-experiment).",
    )
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
    parser.add_argument(
        "--target-tasks",
        type=str,
        default=None,
        help=(
            "Optional comma-separated 1-indexed task IDs to include from the early range. "
            "Tasks before max(target_tasks) that are NOT in this list are skipped; "
            "all tasks from max(target_tasks)+1 up to --task-count are always included."
        ),
    )
    args = parser.parse_args()
    apply_profile_defaults(args)
    if args.target_tasks:
        args.target_tasks = sorted({int(t.strip()) for t in args.target_tasks.split(",") if t.strip()})
    else:
        args.target_tasks = None
    args.device = resolve_collection_device(
        args.device,
        strict_explicit_cuda=bool(str(args.device).strip().startswith("cuda")) if args.device is not None else False,
    )
    args.num_workers = max(1, int(args.num_workers))
    args.gpu_batch_size = max(1, int(args.gpu_batch_size))
    args.gpu_batch_timeout_ms = max(1, int(args.gpu_batch_timeout_ms))
    args.planner_steps = max(1, int(args.planner_steps))
    return args

def _print_cumulative_profile(profiler: cProfile.Profile) -> None:
    """Neatly prints the top 15 time-consumers from the cProfile data."""
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("tottime").print_stats(15)
    print("\nCollection profile (top 15 by total time):")
    print(stream.getvalue().rstrip())

def _seed_output_dir(mode_root: Path, seed_value: int, multi_seed: bool) -> Path:
    """Helper to figure out where to dump the results for a specific seed."""
    return ensure_dir(mode_root / f"seed_{seed_value:04d}") if multi_seed else ensure_dir(mode_root)

class CollectionSupportError(RuntimeError):
    """Raised if we don't get enough successful tasks to form a valid train/val/test split."""
    def __init__(self, message: str, *, dataset_dir: Path, support_check: dict[str, object]):
        super().__init__(message)
        self.dataset_dir = Path(dataset_dir)
        self.support_check = support_check

def _expected_families(task_family: str, family_mix: tuple[tuple[str, float], ...]) -> list[str]:
    """Helper to list all the families we think we should be seeing."""
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
    """
    Checks if we have enough data to actually use this dataset. 
    It counts successful tasks per family and tries to build the splits.
    """
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
    """Converts CLI args into a formal PlannerConfig object."""
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
        repair_max_iter=int(args.repair_max_iter) if args.repair_max_iter is not None else int(args.planner_max_iter or 40),
        repair_max_fun=int(args.repair_max_fun) if args.repair_max_fun is not None else int(args.planner_max_fun or 160),
    )

def _failure_payload(task, alpha: float, restart_index: int, error: Exception) -> dict[str, object]:
    """Pack up a planning failure into a dict for logging."""
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
    """Specifically logs when we skip a job because previous attempts on this task failed."""
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

def collect_one_seed(
    args: argparse.Namespace,
    seed_value: int,
    mode_root: Path,
    alpha_values: list[float],
    multi_seed: bool,
    task_family: str,
    family_mix: tuple[tuple[str, float], ...],
    seed_bank_dir: Path | None = None,
) -> dict[str, object]:
    """
    Main orchestration for collecting data for a single random seed. 
    It samples tasks, runs them (parallel or sequential), and saves everything.
    """
    experiment_dir = _seed_output_dir(mode_root, seed_value=seed_value, multi_seed=multi_seed)
    raw_dir = ensure_dir(experiment_dir / "raw")
    scene_dir = ensure_dir(experiment_dir / "scenes")
    if seed_bank_dir is None:
        seed_bank_dir = mode_root / "seed_bank"
    initial_seed_bank = load_seed_bank(seed_bank_dir, n_waypoints=args.n_waypoints)

    family_config = TaskFamilyConfig(
        task_family=task_family,
        difficulty=args.difficulty,
        benchmark_profile=args.benchmark_profile,
        geometry_regime=args.geometry_regime,
        family_mix=family_mix,
    )
    task_sampler = TaskSampler(seed=seed_value, family_config=family_config)
    tasks = task_sampler.sample_tasks(args.task_count)
    target_tasks: list[int] | None = getattr(args, "target_tasks", None)
    if target_tasks:
        target_set = set(target_tasks)
        max_target = max(target_set)
        tasks = [
            task for task in tasks
            if int(str(task.task_id).split("_")[-1]) in target_set
            or int(str(task.task_id).split("_")[-1]) > max_target
        ]
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
                # Run things one by one on the main thread
                ordered_results = []
                seed_bank_by_family: dict[str, list[SeedEntry]] = {
                    family: list(seeds) for family, seeds in initial_seed_bank.items()
                }
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
                        failure_factory=_failure_payload,
                        skipped_failure_factory=_skipped_failure_payload,
                        mute_stdio_context=_mute_stdio,
                        seed_bank_by_family=seed_bank_by_family,
                    )
                    ordered_results.extend(task_results)
                    for result in task_results:
                        progress_tracker.advance(result)
            else:
                # Use the fancy multiprocessing setup
                ordered_results = _collect_task_results_parallel(
                    tasks,
                    alpha_values=alpha_values,
                    restart_count=int(args.restart_count),
                    scene_dir=scene_dir,
                    planner_mode=args.planner_mode,
                    planner_config=planner_config,
                    num_workers=int(args.num_workers),
                    failure_factory=_failure_payload,
                    skipped_failure_factory=_skipped_failure_payload,
                    mute_stdio_context=_mute_stdio,
                    progress_tracker=progress_tracker,
                    initial_seed_bank=initial_seed_bank,
                    quiet=bool(args.quiet),
                )
    finally:
        progress_tracker.finish()

    ordered_results.sort(key=lambda item: item.order_index)
    records = [item.record for item in ordered_results if item.record is not None]
    failed = [item.failure for item in ordered_results if item.failure is not None]

    # Save the raw records as pickle files
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
    
    # Crunch the numbers for the summary report
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
    
    # Update the seed bank with the best trajectories we found this time
    if deduped:
        persisted_bank: dict[str, list[SeedEntry]] = {}
        for record in deduped:
            family = str(record.get("task_spec", {}).get("family", ""))
            if not family:
                continue
            entry = _seed_entry_from_record(
                record,
                default_task_id=str(record.get("task_spec", {}).get("task_id", "")),
            )
            _maybe_add_family_seed(persisted_bank, family=family, candidate=entry)
        save_seed_bank(persisted_bank, seed_bank_dir, n_waypoints=args.n_waypoints)
    
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
    """The high-level entry point that manages multiple seeds and families."""
    alpha_values = parse_alpha_grid(args.alpha_grid, args.alpha_count, args.alpha_schedule)
    mode_root = ensure_dir(Path(args.dataset_dir)) if args.dataset_dir else ensure_dir(Path(args.output_root) / args.experiment_name / args.planner_mode)
    seed_bank_dir = (
        Path(args.seed_bank_dir) if getattr(args, "seed_bank_dir", None) else mode_root / "seed_bank"
    )
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
                seed_bank_dir=seed_bank_dir,
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
        # Loop through each family and run collection
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
                    seed_bank_dir=seed_bank_dir,
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
    """Entry point! It parses args and kicks off the collection run."""
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

    # If profiling is on, run with cProfile
    profiler = cProfile.Profile()
    try:
        return profiler.runcall(_run_collection, args)
    finally:
        _print_cumulative_profile(profiler)

if __name__ == "__main__":
    main()
