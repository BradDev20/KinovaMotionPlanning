from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from src.morl.run_layout import (
    DEFAULT_RUNS_ROOT,
    checkpoint_dir_for_run,
    compare_dir_for_run,
    dataset_dir_for_run,
    default_compare_output_dir,
    evaluation_dir_for_run,
    pipeline_summary_path_for_run,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")
    return path


def _run_root_path(root: str) -> Path:
    return Path(root)


def _append_option(argv: list[str], flag: str, value) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _append_flag(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if Path(path).exists()]


def _build_collect_backend_argv(args: argparse.Namespace) -> list[str]:
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else dataset_dir_for_run(args.run, args.mode, root=args.root)
    argv: list[str] = [
        "--experiment-name",
        args.run,
        "--planner-mode",
        args.mode,
        "--dataset-dir",
        str(dataset_dir),
        "--task-count",
        str(args.task_count),
        "--restart-count",
        str(args.restart_count),
        "--seed",
        str(args.seed),
        "--output-root",
        str(args.root),
        "--task-family",
        args.family,
        "--difficulty",
        args.difficulty,
        "--benchmark-profile",
        args.benchmark_profile,
        "--geometry-regime",
        args.regime,
        "--n-waypoints",
        str(args.n_waypoints),
        "--cost-sample-rate",
        str(args.cost_sample_rate),
        "--max-steps",
        str(args.max_steps),
        "--alpha-count",
        str(args.alpha_count),
        "--alpha-schedule",
        args.alpha_schedule,
        "--rho",
        str(args.rho),
    ]
    _append_option(argv, "--alpha-grid", args.alpha_grid)
    _append_option(argv, "--seeds", args.seeds)
    _append_option(argv, "--family-mix", args.family_mix)
    _append_option(argv, "--planner-max-iter", args.planner_max_iter)
    _append_option(argv, "--planner-max-fun", args.planner_max_fun)
    _append_option(argv, "--planner-steps", args.planner_steps)
    _append_option(argv, "--repair-max-iter", args.repair_max_iter)
    _append_option(argv, "--repair-max-fun", args.repair_max_fun)
    _append_option(argv, "--device", args.device)
    _append_option(argv, "--num-workers", args.num_workers)
    _append_option(argv, "--gpu-batch-size", args.gpu_batch_size)
    _append_option(argv, "--gpu-batch-timeout-ms", args.gpu_batch_timeout_ms)
    _append_option(argv, "--objective-tol", args.objective_tol)
    _append_option(argv, "--path-tol", args.path_tol)
    _append_option(argv, "--route-tol", args.route_tol)
    _append_flag(argv, "--quiet", bool(getattr(args, "quiet", False)))
    _append_flag(argv, "--profile", bool(getattr(args, "profile", False)))
    _append_flag(argv, "--report-size-matched", args.report_size_matched)
    return argv


def run_collect(args: argparse.Namespace) -> None:
    from src.morl.collect_dataset import main as collect_main

    collect_main(_build_collect_backend_argv(args))


def _resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir)
    if args.run:
        return dataset_dir_for_run(args.run, args.mode, root=args.root)
    raise RuntimeError("Provide --run or --dataset-dir.")


def _resolve_train_output_dir(args: argparse.Namespace, dataset_dir: Path, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    if getattr(args, "checkpoint_dir", None):
        return Path(args.checkpoint_dir)
    if args.run:
        return checkpoint_dir_for_run(args.run, args.mode, root=args.root)
    return dataset_dir / "checkpoints" / f"{args.mode}_iql"


def _build_train_backend_argv(
    args: argparse.Namespace,
    *,
    dataset_dir: Path,
    output_dir: str | Path | None = None,
) -> list[str]:
    resolved_output_dir = _resolve_train_output_dir(args, dataset_dir, output_dir=output_dir)
    argv: list[str] = [
        "--dataset-dir",
        str(dataset_dir),
        "--scalarizer",
        args.mode,
        "--output-dir",
        str(resolved_output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--gamma",
        str(args.gamma),
        "--expectile",
        str(args.expectile),
        "--beta",
        str(args.beta),
        "--rho",
        str(args.rho),
        "--max-joint-velocity",
        str(args.max_joint_velocity),
        "--device",
        args.device or "cpu",
        "--seed",
        str(args.seed),
        "--alpha-conditioning-mode",
        args.alpha_conditioning_mode,
    ]
    _append_option(argv, "--steps-per-epoch", args.steps_per_epoch)
    return argv


def run_train(args: argparse.Namespace) -> None:
    from src.morl.train_offline import main as train_main

    dataset_dir = _resolve_dataset_dir(args)
    _require_path(dataset_dir, label="Dataset directory")
    train_main(_build_train_backend_argv(args, dataset_dir=dataset_dir))


def _resolve_checkpoint_path(
    args: argparse.Namespace,
    *,
    dataset_dir: Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)
    if getattr(args, "checkpoint", None):
        return Path(args.checkpoint)
    if checkpoint_dir is not None:
        return Path(checkpoint_dir) / "checkpoint.pt"
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir) / "checkpoint.pt"
    if getattr(args, "checkpoint_dir", None):
        return Path(args.checkpoint_dir) / "checkpoint.pt"
    if args.run:
        return checkpoint_dir_for_run(args.run, args.mode, root=args.root) / "checkpoint.pt"
    if dataset_dir is None:
        raise RuntimeError("Provide --run or --dataset-dir/--checkpoint.")
    raise RuntimeError("Could not resolve checkpoint path.")


def _resolve_eval_output_dir(
    args: argparse.Namespace,
    *,
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    if getattr(args, "evaluation_dir", None):
        return Path(args.evaluation_dir)
    if args.run:
        return evaluation_dir_for_run(args.run, args.mode, root=args.root)
    from src.morl.run_layout import default_evaluation_output_dir

    return default_evaluation_output_dir(checkpoint_path, dataset_dir=dataset_dir)


def _build_eval_backend_argv(
    args: argparse.Namespace,
    *,
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: str | Path | None = None,
    alpha_grid: str | None = None,
    max_steps: int | None = None,
) -> list[str]:
    resolved_output_dir = _resolve_eval_output_dir(args, dataset_dir=dataset_dir, checkpoint_path=checkpoint_path, output_dir=output_dir)
    resolved_alpha_grid = alpha_grid if alpha_grid is not None else getattr(args, "alpha_grid", None)
    resolved_max_steps = max_steps if max_steps is not None else getattr(args, "max_steps", None)
    argv: list[str] = [
        "--checkpoint",
        str(checkpoint_path),
        "--dataset-dir",
        str(dataset_dir),
        "--scalarizer",
        args.mode,
        "--split",
        args.split,
        "--device",
        args.device or "cpu",
        "--rho",
        str(args.rho),
        "--output-dir",
        str(resolved_output_dir),
    ]
    _append_option(argv, "--alpha-grid", resolved_alpha_grid)
    _append_option(argv, "--max-steps", resolved_max_steps)
    _append_flag(argv, "--deterministic", args.deterministic)
    return argv


def run_eval(args: argparse.Namespace) -> None:
    from src.morl.evaluate import main as evaluate_main

    dataset_dir = _resolve_dataset_dir(args) if args.dataset_dir or args.run else None
    checkpoint_path = _resolve_checkpoint_path(args, dataset_dir=dataset_dir)
    if dataset_dir is None:
        from src.morl.run_layout import infer_dataset_dir_from_checkpoint

        dataset_dir = infer_dataset_dir_from_checkpoint(checkpoint_path)
    _require_path(dataset_dir, label="Dataset directory")
    _require_path(checkpoint_path, label="Checkpoint")
    evaluate_main(_build_eval_backend_argv(args, dataset_dir=dataset_dir, checkpoint_path=checkpoint_path))


def _load_support_check(dataset_dir: Path) -> dict[str, object]:
    dataset_summary_path = dataset_dir / "dataset_summary.json"
    if not dataset_summary_path.exists():
        return {}
    payload = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    support_check = payload.get("support_check")
    return support_check if isinstance(support_check, dict) else {}


def _write_pipeline_summary(
    path: Path,
    *,
    dataset_dir: Path,
    checkpoint_path: Path | None,
    evaluation_dir: Path | None,
    support_check: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    successful_train_tasks = {}
    split_task_ids_by_family = support_check.get("split_task_ids_by_family")
    if isinstance(split_task_ids_by_family, dict):
        successful_train_tasks = split_task_ids_by_family.get("train", {})
    payload = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "evaluation_dir": str(evaluation_dir) if evaluation_dir is not None else None,
        "support_check": support_check,
        "successful_train_task_ids_by_family": successful_train_tasks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_pipeline_result(*, dataset_dir: Path, evaluation_dir: Path | None, support_check: dict[str, object]) -> None:
    print(f"Dataset directory: {dataset_dir}")
    if evaluation_dir is not None:
        print(f"Evaluation directory: {evaluation_dir}")
    split_task_ids_by_family = support_check.get("split_task_ids_by_family")
    train_task_ids_by_family = split_task_ids_by_family.get("train", {}) if isinstance(split_task_ids_by_family, dict) else {}
    print("Successful train-support task IDs by family:")
    print(json.dumps(train_task_ids_by_family, indent=2, sort_keys=True))


def _run_backend_with_optional_stdout_suppression(main_fn, argv: list[str], *, quiet: bool) -> None:
    if not quiet:
        main_fn(argv)
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        main_fn(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    from src.morl.collect_dataset import CollectionSupportError, main as collect_main
    from src.morl.evaluate import main as evaluate_main
    from src.morl.train_offline import main as train_main

    if args.seeds:
        raise RuntimeError("The pipeline command does not support --seeds; run one dataset/training/evaluation flow per seed.")
    if args.family == "all":
        raise RuntimeError("The pipeline command does not support --family all; run one dataset/training/evaluation flow per family.")

    dataset_dir = _resolve_dataset_dir(args)
    checkpoint_dir = _resolve_train_output_dir(args, dataset_dir)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    evaluation_dir = _resolve_eval_output_dir(
        args,
        dataset_dir=dataset_dir,
        checkpoint_path=checkpoint_path,
        output_dir=args.evaluation_dir,
    )
    pipeline_summary_path = pipeline_summary_path_for_run(args.run, args.mode, root=args.root)

    try:
        _run_backend_with_optional_stdout_suppression(
            collect_main,
            _build_collect_backend_argv(args),
            quiet=bool(args.quiet),
        )
    except CollectionSupportError as exc:
        support_check = exc.support_check
        _write_pipeline_summary(
            pipeline_summary_path,
            dataset_dir=exc.dataset_dir,
            checkpoint_path=None,
            evaluation_dir=None,
            support_check=support_check,
        )
        if not bool(args.quiet):
            _print_pipeline_result(dataset_dir=exc.dataset_dir, evaluation_dir=None, support_check=support_check)
        raise

    _require_path(dataset_dir, label="Dataset directory")
    support_check = _load_support_check(dataset_dir)
    if isinstance(support_check, dict) and not bool(support_check.get("passed", True)):
        _write_pipeline_summary(
            pipeline_summary_path,
            dataset_dir=dataset_dir,
            checkpoint_path=None,
            evaluation_dir=None,
            support_check=support_check,
        )
        if not bool(args.quiet):
            _print_pipeline_result(dataset_dir=dataset_dir, evaluation_dir=None, support_check=support_check)
        raise RuntimeError(f"Dataset support check failed: {support_check.get('failure_reason', 'unknown reason')}")

    _run_backend_with_optional_stdout_suppression(
        train_main,
        _build_train_backend_argv(args, dataset_dir=dataset_dir, output_dir=args.checkpoint_dir),
        quiet=bool(args.quiet),
    )
    _run_backend_with_optional_stdout_suppression(
        evaluate_main,
        _build_eval_backend_argv(
            args,
            dataset_dir=dataset_dir,
            checkpoint_path=checkpoint_path,
            output_dir=args.evaluation_dir,
            alpha_grid=args.eval_alpha_grid,
            max_steps=args.eval_max_steps,
        ),
        quiet=bool(args.quiet),
    )
    support_check = _load_support_check(dataset_dir)
    _write_pipeline_summary(
        pipeline_summary_path,
        dataset_dir=dataset_dir,
        checkpoint_path=checkpoint_path,
        evaluation_dir=evaluation_dir,
        support_check=support_check,
    )
    if not bool(args.quiet):
        _print_pipeline_result(dataset_dir=dataset_dir, evaluation_dir=evaluation_dir, support_check=support_check)


def run_compare(args: argparse.Namespace) -> None:
    from src.morl.compare_benchmarks import main as compare_main

    if args.run:
        sum_dir = dataset_dir_for_run(args.run, "sum", root=args.root)
        max_dir = dataset_dir_for_run(args.run, "max", root=args.root)
        output_dir = Path(args.output_dir) if args.output_dir else compare_dir_for_run(args.run, root=args.root)
    else:
        if not args.sum_dir or not args.max_dir:
            raise RuntimeError("Provide either --run or both --sum-dir and --max-dir.")
        sum_dir = Path(args.sum_dir)
        max_dir = Path(args.max_dir)
        output_dir = Path(args.output_dir) if args.output_dir else default_compare_output_dir(sum_dir, max_dir)
    _require_path(sum_dir, label="Weighted-sum dataset")
    _require_path(max_dir, label="Weighted-max dataset")
    argv: list[str] = [
        "--sum-dir",
        str(sum_dir),
        "--max-dir",
        str(max_dir),
        "--output-dir",
        str(output_dir),
        "--size-matched-samples",
        str(args.size_matched_samples),
    ]
    _append_flag(argv, "--allow-profile-mismatch", args.allow_profile_mismatch)
    compare_main(argv)


def run_replay(args: argparse.Namespace) -> None:
    from src.morl.visualize_trajectories import main as replay_main

    planner_dirs = list(args.planner_dirs or [])
    evaluation_rollouts = list(args.evaluation_rollouts or [])
    if args.run and args.mode:
        if args.source in {"both", "planner"}:
            planner_dirs.append(str(dataset_dir_for_run(args.run, args.mode, root=args.root)))
        if args.source in {"both", "eval"}:
            evaluation_rollouts.append(str(evaluation_dir_for_run(args.run, args.mode, root=args.root) / "rollouts.pkl"))
    planner_dirs = _existing_paths(planner_dirs)
    evaluation_rollouts = _existing_paths(evaluation_rollouts)

    task_ids = list(args.tasks or [])
    if args.task:
        task_ids.append(args.task)
    task_ids = [str(task_id) for task_id in task_ids]
    if not task_ids:
        raise RuntimeError("Provide --task or --tasks.")
    argv: list[str] = ["--task-id", *task_ids, "--max-trajectories", str(args.max_trajectories), "--scene-dir", args.scene_dir]
    if not planner_dirs and not evaluation_rollouts:
        raise RuntimeError("No planner or evaluation sources were resolved. Provide --run/--mode or explicit paths.")
    if planner_dirs:
        argv.append("--planner-dirs")
        argv.extend(planner_dirs)
    if evaluation_rollouts:
        argv.append("--evaluation-rollouts")
        argv.extend(evaluation_rollouts)
    _append_option(argv, "--alpha", args.alpha)
    _append_flag(argv, "--no-viewer", args.no_viewer)
    replay_main(argv)


def run_pareto(args: argparse.Namespace) -> None:
    from src.morl.visualize_pareto import main as pareto_main

    planner_dirs = list(args.planner_dirs or [])
    evaluation_dirs = list(args.evaluation_dirs or [])
    if args.run:
        if args.mode:
            planner_dirs.append(str(dataset_dir_for_run(args.run, args.mode, root=args.root)))
            evaluation_dirs.append(str(evaluation_dir_for_run(args.run, args.mode, root=args.root)))
            default_output = evaluation_dir_for_run(args.run, args.mode, root=args.root) / "pareto.png"
        else:
            for mode_name in ("sum", "max"):
                planner_dirs.append(str(dataset_dir_for_run(args.run, mode_name, root=args.root)))
                evaluation_dirs.append(str(evaluation_dir_for_run(args.run, mode_name, root=args.root)))
            default_output = compare_dir_for_run(args.run, root=args.root) / "pareto.png"
    else:
        default_output = Path("data/runs/pareto_comparison.png")
    planner_dirs = _existing_paths(planner_dirs)
    evaluation_dirs = _existing_paths(evaluation_dirs)

    argv: list[str] = [
        "--output",
        str(Path(args.output) if args.output else default_output),
    ]
    if not planner_dirs and not evaluation_dirs:
        raise RuntimeError("No planner or evaluation data found for Pareto plotting.")
    if planner_dirs:
        argv.append("--planner-dirs")
        argv.extend(planner_dirs)
    if evaluation_dirs:
        argv.append("--evaluation-dirs")
        argv.extend(evaluation_dirs)
    if args.summary_output:
        argv.extend(["--summary-output", args.summary_output])
    argv.extend(["--source", args.source])
    _append_flag(argv, "--group-by-family", args.group_by_family)
    _append_flag(argv, "--coverage-only", args.coverage_only)
    _append_flag(argv, "--color-by-alpha", args.color_by_alpha)
    _append_flag(argv, "--nonconvex-only", args.nonconvex_only)
    pareto_main(argv)


def _run_gen3_example(*, viewer: bool, steps: int) -> None:
    from src.examples.gen3 import gen3_env

    env = gen3_env(headless=not viewer)
    env.start_viewer()
    try:
        for _ in range(max(steps, 1)):
            env.step()
            if viewer:
                time.sleep(0.01)
    finally:
        env.close()
    print(f"Completed {steps} gen3 simulation steps.")


def run_check(args: argparse.Namespace) -> None:
    if args.viewer:
        _run_gen3_example(viewer=bool(args.viewer), steps=args.steps)
        return

    import mujoco

    scene_path = _repo_root() / "robot_models" / "kinova_gen3" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    mujoco.MjData(model)
    print("MuJoCo import: OK")
    print(f"Robot model load: OK ({scene_path})")
    print("Environment check completed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified MotionPlanning CLI for dataset collection, training, evaluation, replay, and onboarding."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect a planner dataset into the flattened run layout.")
    collect_parser.add_argument("--run", required=True, help="Run name, e.g. bench_culdesac_t30_a11_r5_20260413.")
    collect_parser.add_argument("--mode", choices=["sum", "max"], required=True, help="Planner mode.")
    collect_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    collect_parser.add_argument("--dataset-dir", default=None, help="Optional explicit dataset directory.")
    collect_parser.add_argument("--family", default="mixed", help="Task family.")
    collect_parser.add_argument("--regime", choices=["mixed", "convex", "nonconvex"], default="mixed", help="Geometry regime.")
    collect_parser.add_argument("--benchmark-profile", choices=["baseline", "max_favoring"], default="baseline", help="Benchmark profile.")
    collect_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium", help="Task difficulty.")
    collect_parser.add_argument("--task-count", type=int, default=30, help="Number of tasks to sample.")
    collect_parser.add_argument("--alpha-grid", type=str, default=None, help="Optional comma-separated alpha values.")
    collect_parser.add_argument("--alpha-count", type=int, default=11, help="Number of alpha values when --alpha-grid is omitted.")
    collect_parser.add_argument("--alpha-schedule", choices=["linear", "dense-middle", "dense-ends"], default="linear", help="Alpha schedule when --alpha-grid is omitted.")
    collect_parser.add_argument("--restart-count", type=int, default=5, help="Planner restarts per task/alpha.")
    collect_parser.add_argument("--seed", type=int, default=0, help="Single collection seed.")
    collect_parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated collection seeds.")
    collect_parser.add_argument("--family-mix", type=str, default=None, help="Optional explicit family mixture.")
    collect_parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-break parameter.")
    collect_parser.add_argument("--n-waypoints", type=int, default=25, help="Planner waypoint count.")
    collect_parser.add_argument("--cost-sample-rate", type=int, default=2, help="Planner cost sampling stride.")
    collect_parser.add_argument("--planner-max-iter", type=int, default=None, help="Optional planner iteration cap.")
    collect_parser.add_argument("--planner-max-fun", type=int, default=None, help="Optional planner function-eval cap.")
    collect_parser.add_argument("--planner-steps", type=int, default=250, help="Torch optimizer steps per batch.")
    collect_parser.add_argument("--repair-max-iter", type=int, default=None, help="Optional CPU repair iteration cap.")
    collect_parser.add_argument("--repair-max-fun", type=int, default=None, help="Optional CPU repair function-eval cap.")
    collect_parser.add_argument("--device", default=None, help="Collection planner device.")
    collect_parser.add_argument("--num-workers", type=int, default=None, help="CPU task worker count.")
    collect_parser.add_argument("--gpu-batch-size", type=int, default=32, help="Torch planner micro-batch size.")
    collect_parser.add_argument("--gpu-batch-timeout-ms", type=int, default=10, help="Torch planner micro-batch timeout in milliseconds.")
    collect_parser.add_argument("--quiet", action="store_true", help="Suppress non-progress collector output.")
    collect_parser.add_argument("--max-steps", type=int, default=25, help="Offline RL horizon for transition generation.")
    collect_parser.add_argument("--objective-tol", type=float, default=1e-3, help="Objective tolerance for deduplication.")
    collect_parser.add_argument("--path-tol", type=float, default=5e-2, help="Path tolerance for deduplication.")
    collect_parser.add_argument("--route-tol", type=float, default=None, help="Optional route clustering tolerance.")
    collect_parser.add_argument("--profile", action="store_true", help="Print a cumulative cProfile report for collection execution.")
    collect_parser.add_argument("--report-size-matched", action="store_true", help="Mark the run for downstream size-matched reporting.")
    collect_parser.set_defaults(func=run_collect)

    pipeline_parser = subparsers.add_parser("pipeline", help="Collect, train, and evaluate in one support-aware workflow.")
    pipeline_parser.add_argument("--run", required=True, help="Run name, e.g. bench_culdesac_t30_a11_r5_20260413.")
    pipeline_parser.add_argument("--mode", choices=["sum", "max"], required=True, help="Planner/training/evaluation mode.")
    pipeline_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    pipeline_parser.add_argument("--dataset-dir", default=None, help="Optional explicit dataset directory.")
    pipeline_parser.add_argument("--family", default="mixed", help="Task family.")
    pipeline_parser.add_argument("--regime", choices=["mixed", "convex", "nonconvex"], default="mixed", help="Geometry regime.")
    pipeline_parser.add_argument("--benchmark-profile", choices=["baseline", "max_favoring"], default="baseline", help="Benchmark profile.")
    pipeline_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium", help="Task difficulty.")
    pipeline_parser.add_argument("--task-count", type=int, default=30, help="Number of tasks to sample.")
    pipeline_parser.add_argument("--alpha-grid", type=str, default=None, help="Optional comma-separated collection alpha values.")
    pipeline_parser.add_argument("--alpha-count", type=int, default=11, help="Number of collection alpha values when --alpha-grid is omitted.")
    pipeline_parser.add_argument("--alpha-schedule", choices=["linear", "dense-middle", "dense-ends"], default="linear", help="Collection alpha schedule when --alpha-grid is omitted.")
    pipeline_parser.add_argument("--restart-count", type=int, default=5, help="Planner restarts per task/alpha.")
    pipeline_parser.add_argument("--seed", type=int, default=0, help="Single collection/training seed.")
    pipeline_parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated collection seeds. Not supported by pipeline.")
    pipeline_parser.add_argument("--family-mix", type=str, default=None, help="Optional explicit family mixture.")
    pipeline_parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-break parameter.")
    pipeline_parser.add_argument("--n-waypoints", type=int, default=25, help="Planner waypoint count.")
    pipeline_parser.add_argument("--cost-sample-rate", type=int, default=2, help="Planner cost sampling stride.")
    pipeline_parser.add_argument("--planner-max-iter", type=int, default=None, help="Optional planner iteration cap.")
    pipeline_parser.add_argument("--planner-max-fun", type=int, default=None, help="Optional planner function-eval cap.")
    pipeline_parser.add_argument("--planner-steps", type=int, default=250, help="Torch optimizer steps per batch.")
    pipeline_parser.add_argument("--repair-max-iter", type=int, default=None, help="Optional CPU repair iteration cap.")
    pipeline_parser.add_argument("--repair-max-fun", type=int, default=None, help="Optional CPU repair function-eval cap.")
    pipeline_parser.add_argument("--num-workers", type=int, default=None, help="CPU task worker count.")
    pipeline_parser.add_argument("--gpu-batch-size", type=int, default=32, help="Torch planner micro-batch size.")
    pipeline_parser.add_argument("--gpu-batch-timeout-ms", type=int, default=10, help="Torch planner micro-batch timeout in milliseconds.")
    pipeline_parser.add_argument("--quiet", action="store_true", help="Suppress non-progress pipeline output.")
    pipeline_parser.add_argument("--max-steps", type=int, default=25, help="Offline RL horizon for transition generation.")
    pipeline_parser.add_argument("--objective-tol", type=float, default=1e-3, help="Objective tolerance for deduplication.")
    pipeline_parser.add_argument("--path-tol", type=float, default=5e-2, help="Path tolerance for deduplication.")
    pipeline_parser.add_argument("--route-tol", type=float, default=None, help="Optional route clustering tolerance.")
    pipeline_parser.add_argument("--profile", action="store_true", help="Print a cumulative cProfile report for collection execution.")
    pipeline_parser.add_argument("--report-size-matched", action="store_true", help="Mark the run for downstream size-matched reporting.")
    pipeline_parser.add_argument("--checkpoint-dir", default=None, help="Optional explicit checkpoint directory.")
    pipeline_parser.add_argument("--alpha-conditioning-mode", choices=["dataset", "uniform"], default="dataset", help="Alpha conditioning mode.")
    pipeline_parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    pipeline_parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    pipeline_parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional training steps per epoch.")
    pipeline_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width.")
    pipeline_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    pipeline_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    pipeline_parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile.")
    pipeline_parser.add_argument("--beta", type=float, default=3.0, help="IQL actor temperature.")
    pipeline_parser.add_argument("--max-joint-velocity", type=float, default=1.3, help="Policy action cap from joint velocity.")
    pipeline_parser.add_argument("--device", default=None, help="Torch device.")
    pipeline_parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate.")
    pipeline_parser.add_argument("--eval-alpha-grid", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated evaluation alpha values.")
    pipeline_parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    pipeline_parser.add_argument("--eval-max-steps", type=int, default=None, help="Optional evaluation horizon override.")
    pipeline_parser.add_argument("--evaluation-dir", default=None, help="Optional explicit evaluation output directory.")
    pipeline_parser.set_defaults(func=run_pipeline)

    train_parser = subparsers.add_parser("train", help="Train an offline policy from a flattened or legacy dataset.")
    train_parser.add_argument("--run", default=None, help="Run name.")
    train_parser.add_argument("--mode", choices=["sum", "max"], required=True, help="Training mode.")
    train_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    train_parser.add_argument("--dataset-dir", default=None, help="Optional explicit dataset directory.")
    train_parser.add_argument("--output-dir", default=None, help="Optional explicit checkpoint directory.")
    train_parser.add_argument("--alpha-conditioning-mode", choices=["dataset", "uniform"], default="dataset", help="Alpha conditioning mode.")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    train_parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional training steps per epoch.")
    train_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width.")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    train_parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile.")
    train_parser.add_argument("--beta", type=float, default=3.0, help="IQL actor temperature.")
    train_parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-break parameter.")
    train_parser.add_argument("--max-joint-velocity", type=float, default=1.3, help="Policy action cap from joint velocity.")
    train_parser.add_argument("--device", default="cpu", help="Torch device.")
    train_parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    train_parser.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy using flattened run defaults.")
    eval_parser.add_argument("--run", default=None, help="Run name.")
    eval_parser.add_argument("--mode", choices=["sum", "max"], required=True, help="Evaluation mode.")
    eval_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    eval_parser.add_argument("--dataset-dir", default=None, help="Optional explicit dataset directory.")
    eval_parser.add_argument("--checkpoint-dir", default=None, help="Optional explicit checkpoint directory.")
    eval_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate.")
    eval_parser.add_argument("--alpha-grid", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated alpha values.")
    eval_parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-break parameter.")
    eval_parser.add_argument("--device", default="cpu", help="Torch device.")
    eval_parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    eval_parser.add_argument("--max-steps", type=int, default=None, help="Optional evaluation horizon override.")
    eval_parser.add_argument("--output-dir", default=None, help="Optional explicit evaluation output directory.")
    eval_parser.set_defaults(func=run_eval)

    compare_parser = subparsers.add_parser("compare", help="Compare sum and max planner datasets.")
    compare_parser.add_argument("--run", default=None, help="Run name. When provided, sum/max dataset paths are inferred.")
    compare_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    compare_parser.add_argument("--sum-dir", default=None, help="Explicit weighted-sum dataset directory.")
    compare_parser.add_argument("--max-dir", default=None, help="Explicit weighted-max dataset directory.")
    compare_parser.add_argument("--output-dir", default=None, help="Optional explicit comparison output directory.")
    compare_parser.add_argument("--size-matched-samples", type=int, default=100, help="Bootstrap samples for size-matched max ablation.")
    compare_parser.add_argument("--allow-profile-mismatch", action="store_true", help="Allow comparing datasets with mismatched profile/regime metadata.")
    compare_parser.set_defaults(func=run_compare)

    replay_parser = subparsers.add_parser("replay", help="Replay planner and/or policy rollouts for a task.")
    replay_parser.add_argument("--run", default=None, help="Run name.")
    replay_parser.add_argument("--mode", choices=["sum", "max"], default=None, help="Mode to replay when --run is provided.")
    replay_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    replay_parser.add_argument("--task", default=None, help="Single task id to replay.")
    replay_parser.add_argument("--tasks", nargs="+", default=None, help="One or more task ids to replay in sequence.")
    replay_parser.add_argument("--alpha", type=float, default=None, help="Optional alpha selector.")
    replay_parser.add_argument("--source", choices=["both", "planner", "eval"], default="both", help="Replay source selection when using --run.")
    replay_parser.add_argument("--planner-dirs", nargs="*", default=None, help="Explicit planner dataset directories.")
    replay_parser.add_argument("--evaluation-rollouts", nargs="*", default=None, help="Explicit evaluation rollout pickle files.")
    replay_parser.add_argument("--max-trajectories", type=int, default=4, help="Maximum trajectories to replay.")
    replay_parser.add_argument("--scene-dir", default="data/runs/_visualization/scenes", help="Temporary overlay scene directory.")
    replay_parser.add_argument("--no-viewer", action="store_true", help="Print selected trajectories without launching the viewer.")
    replay_parser.set_defaults(func=run_replay)

    pareto_parser = subparsers.add_parser("pareto", help="Plot planner and evaluation Pareto fronts.")
    pareto_parser.add_argument("--run", default=None, help="Run name.")
    pareto_parser.add_argument("--mode", choices=["sum", "max"], default=None, help="Optional single mode for run-scoped plotting.")
    pareto_parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT), help="Run root directory.")
    pareto_parser.add_argument("--planner-dirs", nargs="*", default=None, help="Explicit planner dataset directories.")
    pareto_parser.add_argument("--evaluation-dirs", nargs="*", default=None, help="Explicit evaluation directories.")
    pareto_parser.add_argument("--source", choices=["both", "planner", "offline_rl"], default="both", help="Select whether to show planner data, offline RL data, or both.")
    pareto_parser.add_argument("--output", default=None, help="Output PNG path.")
    pareto_parser.add_argument("--summary-output", default=None, help="Optional JSON summary output.")
    pareto_parser.add_argument("--group-by-family", action="store_true", help="Create small multiples by task family.")
    pareto_parser.add_argument("--coverage-only", action="store_true", help="Plot Pareto-front points only.")
    pareto_parser.add_argument("--color-by-alpha", action="store_true", help="Color planner points by alpha.")
    pareto_parser.add_argument("--nonconvex-only", action="store_true", help="Plot only non-convex families.")
    pareto_parser.set_defaults(func=run_pareto)

    check_parser = subparsers.add_parser("check", help="Run an environment/model check.")
    check_parser.add_argument("--viewer", action="store_true", help="Launch the interactive Gen3 viewer check.")
    check_parser.add_argument("--steps", type=int, default=200, help="Simulation steps for the Gen3 viewer check.")
    check_parser.set_defaults(func=run_check)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
