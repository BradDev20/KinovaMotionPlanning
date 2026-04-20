from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

from .config import EnvConfig
from .dataset import load_split_manifest
from .iql import PreferenceConditionedIQL, require_torch, torch
from .planning import _plasma_color
from .run_layout import default_evaluation_output_dir, infer_dataset_dir_from_checkpoint
from .scalarization import alpha_to_weights, scalarize_numpy
from .semantics import (
    ARTIFACT_SEMANTICS_VERSION,
    artifact_semantics_payload,
    require_current_artifact_semantics,
    semantics_version_from_payload,
)
from .tasks import load_tasks


def parse_alpha_grid(alpha_grid: str) -> list[float]:
    return [float(token.strip()) for token in alpha_grid.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MORL checkpoints on held-out Kinova tasks.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint.pt produced by training.")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory. Defaults to checkpoint parent parents.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Held-out task split.")
    parser.add_argument("--alpha-grid", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated alpha values.")
    parser.add_argument("--scalarizer", choices=["sum", "max"], required=True, help="Evaluation scalarizer for reporting.")
    parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-breaking parameter.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument("--max-steps", type=int, default=None, help="Evaluation horizon. Defaults to planner waypoint count - 1 when available.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for evaluation outputs.")
    return parser.parse_args()


def infer_max_steps(dataset_dir: Path, tasks: list) -> int:
    experiment_config_path = dataset_dir / "experiment_config.json"
    if experiment_config_path.exists():
        payload = json.loads(experiment_config_path.read_text(encoding="utf-8"))
        planner_waypoints = int(payload.get("planner_config", {}).get("n_waypoints", 0))
        if planner_waypoints > 1:
            return planner_waypoints - 1
    return max(int(task.horizon) for task in tasks) if tasks else 25


def main(argv: list[str] | None = None) -> None:
    import sys

    if argv is not None:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], *argv]
    try:
        args = parse_args()
    finally:
        if argv is not None:
            sys.argv = original_argv
    require_torch()
    from .env import KinovaMORLEnv

    checkpoint_path = Path(args.checkpoint)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else infer_dataset_dir_from_checkpoint(checkpoint_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_evaluation_output_dir(checkpoint_path, dataset_dir=dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = require_current_artifact_semantics(
        dataset_dir / "dataset_summary.json",
        artifact_label="Dataset summary",
    )
    payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    checkpoint_version = semantics_version_from_payload(payload)
    if checkpoint_version != ARTIFACT_SEMANTICS_VERSION:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} uses artifact semantics '{checkpoint_version}'. "
            f"Expected '{ARTIFACT_SEMANTICS_VERSION}'. Retrain the policy on a rebuilt dataset before evaluation."
        )
    agent = PreferenceConditionedIQL.from_checkpoint(payload, device=args.device)
    tasks = load_tasks(dataset_dir / "tasks.json")
    task_by_id = {task.task_id: task for task in tasks}
    split_manifest = load_split_manifest(dataset_dir / "splits.json")
    selected_task_ids = split_manifest.get(args.split, [])
    alpha_values = parse_alpha_grid(args.alpha_grid)
    max_steps = int(args.max_steps) if args.max_steps is not None else infer_max_steps(dataset_dir, tasks)
    scene_dir = output_dir / "scenes"
    rollouts = []

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "alpha",
                "length_weight",
                "obstacle_weight",
                "length_objective",
                "obstacle_objective",
                "scalarized_objective",
                "steps",
                "success",
                "collision",
                "timeout",
                "final_error",
            ]
        )

        for task_id in selected_task_ids:
            task = task_by_id[task_id]
            for alpha in alpha_values:
                weights = alpha_to_weights(alpha)
                env = KinovaMORLEnv(task, scene_dir=scene_dir, env_config=EnvConfig(max_steps=max_steps))
                rollout = env.rollout(agent.act, weights, deterministic=args.deterministic)
                scalarized = float(
                    scalarize_numpy(
                        rollout["objective_total"][None, :],
                        weights,
                        args.scalarizer,
                        rho=args.rho,
                    )[0]
                )
                writer.writerow(
                    [
                        task_id,
                        alpha,
                        float(weights[0]),
                        float(weights[1]),
                        float(rollout["objective_total"][0]),
                        float(rollout["objective_total"][1]),
                        scalarized,
                        rollout["steps"],
                        rollout["success"],
                        rollout["collision"],
                        rollout["timeout"],
                        rollout["final_error"],
                    ]
                )
                rollouts.append(
                    {
                        "trajectory_id": f"eval_{task_id}_{args.scalarizer}_a{alpha:.3f}".replace(".", "p"),
                        "task_id": task_id,
                        "trajectory": rollout["trajectory"],
                        "alpha": float(alpha),
                        "length_weight": float(weights[0]),
                        "obstacle_weight": float(weights[1]),
                        "length_cost": float(rollout["objective_total"][0]),
                        "obstacle_cost": float(rollout["objective_total"][1]),
                        "scalarized_cost": scalarized,
                        "waypoint_count": int(rollout["trajectory"].shape[0]),
                        "color": _plasma_color(alpha),
                        "timestamp": None,
                        "task_spec": rollout["task_spec"],
                        "source": "offline_rl",
                        "scalarizer": args.scalarizer,
                        "success": bool(rollout["success"]),
                        "collision": bool(rollout["collision"]),
                        "timeout": bool(rollout["timeout"]),
                        "final_error": float(rollout["final_error"]),
                    }
                )

    with (output_dir / "rollouts.pkl").open("wb") as handle:
        pickle.dump(rollouts, handle)
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "alpha_grid": alpha_values,
        "scalarizer": args.scalarizer,
        "rollout_count": len(rollouts),
        "max_steps": max_steps,
        "dataset_artifact_semantics_version": str(dataset_summary["artifact_semantics_version"]),
        **artifact_semantics_payload(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {csv_path}")


if __name__ == "__main__":
    main()
