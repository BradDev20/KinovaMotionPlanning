from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_seed_values(seeds: str | None, seed_count: int, seed_start: int) -> list[int]:
    if seeds:
        values = [int(token.strip()) for token in str(seeds).split(",") if token.strip()]
        if not values:
            raise RuntimeError("No valid seed values were provided in --seeds.")
        return values
    count = max(int(seed_count), 1)
    start = int(seed_start)
    return [start + index for index in range(count)]


def _default_checkpoint_root(dataset_dir: Path) -> Path:
    if dataset_dir.name == "dataset":
        return dataset_dir.parent / "checkpoints"
    return dataset_dir / "checkpoints"


def _default_evaluation_root(dataset_dir: Path) -> Path:
    if dataset_dir.name == "dataset":
        return dataset_dir.parent / "evaluation"
    return dataset_dir / "evaluation"


def _train_artifacts_ready(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "checkpoint.pt").exists() and (checkpoint_dir / "summary.json").exists()


def _eval_artifacts_ready(evaluation_dir: Path) -> bool:
    return (evaluation_dir / "metrics.csv").exists() and (evaluation_dir / "summary.json").exists()


def _read_eval_metrics(evaluation_dir: Path) -> dict[str, float]:
    rows = list(csv.DictReader((evaluation_dir / "metrics.csv").open("r", newline="", encoding="utf-8")))
    rollout_count = len(rows)
    if rollout_count == 0:
        return {
            "rollout_count": 0.0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "mean_length_objective": 0.0,
            "mean_obstacle_objective": 0.0,
            "mean_scalarized_objective": 0.0,
        }

    success_count = sum(1 for row in rows if str(row["success"]).lower() == "true")
    collision_count = sum(1 for row in rows if str(row["collision"]).lower() == "true")
    timeout_count = sum(1 for row in rows if str(row["timeout"]).lower() == "true")
    length_total = sum(float(row["length_objective"]) for row in rows)
    obstacle_total = sum(float(row["obstacle_objective"]) for row in rows)
    scalarized_total = sum(float(row["scalarized_objective"]) for row in rows)
    denominator = float(rollout_count)
    return {
        "rollout_count": float(rollout_count),
        "success_rate": float(success_count) / denominator,
        "collision_rate": float(collision_count) / denominator,
        "timeout_rate": float(timeout_count) / denominator,
        "mean_length_objective": length_total / denominator,
        "mean_obstacle_objective": obstacle_total / denominator,
        "mean_scalarized_objective": scalarized_total / denominator,
    }


def _aggregate_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-dataset train/eval seed sweep and summarize stability.")
    parser.add_argument("--dataset-dir", required=True, help="Fixed dataset directory used for all seeds.")
    parser.add_argument("--mode", choices=["sum", "max"], required=True, help="Scalarizer/mode for train + eval.")
    parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated seed values.")
    parser.add_argument("--seed-count", type=int, default=5, help="Number of seeds when --seeds is omitted.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed when --seeds is omitted.")
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Root directory for per-seed checkpoints.")
    parser.add_argument("--evaluation-root", type=str, default=None, help="Root directory for per-seed evaluations.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Evaluation split.")
    parser.add_argument("--alpha-grid", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Evaluation alpha grid.")
    parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-break parameter.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for train/eval.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional evaluation horizon override.")
    parser.add_argument("--alpha-conditioning-mode", choices=["dataset", "uniform"], default="dataset", help="Alpha conditioning mode.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional steps per epoch.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Training hidden layer width.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Training learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile.")
    parser.add_argument("--beta", type=float, default=3.0, help="IQL actor temperature.")
    parser.add_argument("--max-joint-velocity", type=float, default=1.3, help="Maximum joint velocity used for policy action cap.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic evaluation policy actions.")
    parser.add_argument("--stochastic", action="store_true", help="Disable deterministic evaluation policy actions.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip train/eval work for seeds with existing artifacts.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Force rerun even if artifacts already exist.")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--summary-csv", type=str, default=None, help="Optional per-seed summary CSV path.")
    parser.set_defaults(deterministic=True, skip_existing=True)
    return parser.parse_args()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    import sys

    if argv is not None:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], *argv]
    try:
        args = parse_args()
    finally:
        if argv is not None:
            sys.argv = original_argv

    from .evaluate import main as evaluate_main
    from .train_offline import main as train_main

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {dataset_dir}")

    deterministic = bool(args.deterministic) and not bool(args.stochastic)
    skip_existing = bool(args.skip_existing) and not bool(args.no_skip_existing)
    seed_values = _parse_seed_values(args.seeds, seed_count=args.seed_count, seed_start=args.seed_start)

    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else _default_checkpoint_root(dataset_dir)
    evaluation_root = Path(args.evaluation_root) if args.evaluation_root else _default_evaluation_root(dataset_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        checkpoint_dir = checkpoint_root / f"{args.mode}_iql_seed_{seed:04d}"
        evaluation_dir = evaluation_root / f"seed_{seed:04d}"

        if not (skip_existing and _train_artifacts_ready(checkpoint_dir)):
            train_argv: list[str] = [
                "--dataset-dir",
                str(dataset_dir),
                "--scalarizer",
                args.mode,
                "--output-dir",
                str(checkpoint_dir),
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
                str(args.device),
                "--seed",
                str(seed),
                "--alpha-conditioning-mode",
                str(args.alpha_conditioning_mode),
            ]
            if args.steps_per_epoch is not None:
                train_argv.extend(["--steps-per-epoch", str(args.steps_per_epoch)])
            train_main(train_argv)

        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint missing after training: {checkpoint_path}")

        if not (skip_existing and _eval_artifacts_ready(evaluation_dir)):
            eval_argv: list[str] = [
                "--checkpoint",
                str(checkpoint_path),
                "--dataset-dir",
                str(dataset_dir),
                "--split",
                str(args.split),
                "--alpha-grid",
                str(args.alpha_grid),
                "--scalarizer",
                args.mode,
                "--rho",
                str(args.rho),
                "--device",
                str(args.device),
                "--output-dir",
                str(evaluation_dir),
            ]
            if deterministic:
                eval_argv.append("--deterministic")
            if args.max_steps is not None:
                eval_argv.extend(["--max-steps", str(args.max_steps)])
            evaluate_main(eval_argv)

        metrics = _read_eval_metrics(evaluation_dir)
        per_seed_rows.append(
            {
                "seed": int(seed),
                "checkpoint_dir": str(checkpoint_dir),
                "evaluation_dir": str(evaluation_dir),
                **metrics,
            }
        )

    aggregate = {
        metric: _aggregate_metric([float(row[metric]) for row in per_seed_rows])
        for metric in (
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_length_objective",
            "mean_obstacle_objective",
            "mean_scalarized_objective",
        )
    }
    aggregate["seed_count"] = len(per_seed_rows)

    summary_payload: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "mode": str(args.mode),
        "split": str(args.split),
        "alpha_grid": str(args.alpha_grid),
        "deterministic": deterministic,
        "skip_existing": skip_existing,
        "seed_values": [int(seed) for seed in seed_values],
        "checkpoint_root": str(checkpoint_root),
        "evaluation_root": str(evaluation_root),
        "per_seed": per_seed_rows,
        "aggregate": aggregate,
    }

    summary_json_path = Path(args.summary_json) if args.summary_json else evaluation_root / "seed_sweep_summary.json"
    summary_csv_path = Path(args.summary_csv) if args.summary_csv else evaluation_root / "seed_sweep_summary.csv"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "seed",
            "checkpoint_dir",
            "evaluation_dir",
            "rollout_count",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_length_objective",
            "mean_obstacle_objective",
            "mean_scalarized_objective",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({name: row[name] for name in fieldnames})

    print(f"Saved seed sweep summary to {summary_json_path}")
    print(f"Saved per-seed summary CSV to {summary_csv_path}")
    return summary_payload


if __name__ == "__main__":
    main()
