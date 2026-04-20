from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .dataset import load_split_manifest, load_transition_dataset
from .iql import IQLConfig, PreferenceConditionedIQL, require_torch, torch
from .run_layout import default_training_output_dir
from .scalarization import alpha_to_weights
from .semantics import (
    ARTIFACT_SEMANTICS_VERSION,
    artifact_semantics_payload,
    require_current_artifact_semantics,
)
from .tasks import load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train preference-conditioned offline MORL with IQL-lite.")
    parser.add_argument("--dataset-dir", required=True, help="Directory produced by src.morl.collect_dataset.")
    parser.add_argument("--scalarizer", choices=["sum", "max"], required=True, help="Training scalarizer mode.")
    parser.add_argument(
        "--alpha-conditioning-mode",
        choices=["dataset", "uniform"],
        default="dataset",
        help="Use trajectory weights from the dataset or resample alphas uniformly each batch.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional gradient steps per epoch.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--expectile", type=float, default=0.7, help="Expectile parameter.")
    parser.add_argument("--beta", type=float, default=3.0, help="IQL actor temperature.")
    parser.add_argument("--rho", type=float, default=0.01, help="Weighted-max tie-breaking parameter.")
    parser.add_argument("--max-joint-velocity", type=float, default=1.3, help="Maximum joint velocity used to cap policy action deltas.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint directory.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    require_torch()
    torch.manual_seed(seed)


def subset_dataset(dataset: dict[str, np.ndarray], split_ids: list[str]) -> dict[str, np.ndarray]:
    if not split_ids:
        return {key: value[:0] for key, value in dataset.items()}
    mask = np.isin(dataset["task_ids"], np.asarray(split_ids))
    return {key: value[mask] for key, value in dataset.items()}


def sample_batch(dataset: dict[str, np.ndarray], batch_size: int, conditioning_mode: str) -> dict[str, np.ndarray]:
    indices = np.random.randint(0, dataset["observations"].shape[0], size=batch_size)
    batch = {key: value[indices] for key, value in dataset.items()}
    if conditioning_mode == "uniform":
        alpha_samples = np.random.uniform(low=0.0, high=1.0, size=batch_size)
        batch["weights"] = np.stack([alpha_to_weights(alpha) for alpha in alpha_samples], axis=0).astype(np.float32)
    return batch


def compute_obs_stats(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mean = dataset["observations"].mean(axis=0).astype(np.float32)
    std = dataset["observations"].std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def infer_action_limit(dataset_dir: Path, dataset: dict[str, np.ndarray], max_joint_velocity: float) -> float:
    max_observed_action = float(np.max(np.abs(dataset["actions"]))) if dataset["actions"].size else 0.0
    tasks = load_tasks(dataset_dir / "tasks.json")
    if not tasks:
        return max(max_observed_action, 1e-3)
    min_dt = min(float(task.dt) for task in tasks)
    physical_limit = float(max_joint_velocity) * float(min_dt)
    if max_observed_action <= 0.0:
        return max(physical_limit, 1e-3)
    return max(min(max_observed_action, physical_limit), 1e-3)


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
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    dataset_summary = require_current_artifact_semantics(
        dataset_dir / "dataset_summary.json",
        artifact_label="Dataset summary",
    )
    support_check = dataset_summary.get("support_check")
    if isinstance(support_check, dict) and not bool(support_check.get("passed", True)):
        raise RuntimeError(f"Dataset support check failed: {support_check.get('failure_reason', 'unknown reason')}")
    transitions = load_transition_dataset(dataset_dir / "transitions.npz")
    splits = load_split_manifest(dataset_dir / "splits.json")
    train_data = subset_dataset(transitions, splits.get("train", []))
    val_data = subset_dataset(transitions, splits.get("val", []))
    if train_data["observations"].shape[0] == 0:
        raise RuntimeError("Training split is empty. Increase task count or adjust splits.")

    obs_mean, obs_std = compute_obs_stats(train_data)
    action_limit = infer_action_limit(dataset_dir, train_data, max_joint_velocity=args.max_joint_velocity)
    config = IQLConfig(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        expectile=args.expectile,
        beta=args.beta,
        rho=args.rho,
        scalarizer_mode=args.scalarizer,
        action_limit=action_limit,
    )
    agent = PreferenceConditionedIQL(
        obs_dim=train_data["observations"].shape[1],
        action_dim=train_data["actions"].shape[1],
        config=config,
        device=args.device,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

    steps_per_epoch = args.steps_per_epoch or max(train_data["observations"].shape[0] // args.batch_size, 1)
    output_dir = Path(args.output_dir) if args.output_dir else default_training_output_dir(dataset_dir, args.scalarizer)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        metrics_accumulator = {"q_loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0, "advantage_mean": 0.0}
        for _ in range(steps_per_epoch):
            batch = sample_batch(train_data, args.batch_size, args.alpha_conditioning_mode)
            step_metrics = agent.train_step(batch)
            for key, value in step_metrics.items():
                metrics_accumulator[key] += value

        epoch_metrics = {key: value / steps_per_epoch for key, value in metrics_accumulator.items()}
        epoch_metrics["epoch"] = epoch + 1
        if val_data["observations"].shape[0] > 0:
            val_batch = sample_batch(val_data, min(args.batch_size, val_data["observations"].shape[0]), "dataset")
            with torch.no_grad():
                normalized_obs = agent._normalize_obs(torch.as_tensor(val_batch["observations"], dtype=torch.float32, device=agent.device))
                normalized_next_obs = agent._normalize_obs(torch.as_tensor(val_batch["next_observations"], dtype=torch.float32, device=agent.device))
                val_actions = torch.as_tensor(val_batch["actions"], dtype=torch.float32, device=agent.device)
                val_rewards = torch.as_tensor(val_batch["reward_vectors"], dtype=torch.float32, device=agent.device)
                val_dones = torch.as_tensor(val_batch["dones"], dtype=torch.float32, device=agent.device).unsqueeze(-1)
                q_target = val_rewards + agent.config.gamma * (1.0 - val_dones) * agent.value(normalized_next_obs)
                q_pred = torch.minimum(agent.q1(normalized_obs, val_actions), agent.q2(normalized_obs, val_actions))
                epoch_metrics["val_q_loss"] = float(torch.nn.functional.mse_loss(q_pred, q_target).item())
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"q={epoch_metrics['q_loss']:.4f} v={epoch_metrics['value_loss']:.4f} pi={epoch_metrics['policy_loss']:.4f}"
        )

    checkpoint = agent.checkpoint_payload()
    checkpoint["training_args"] = vars(args)
    checkpoint["artifact_semantics_version"] = ARTIFACT_SEMANTICS_VERSION
    checkpoint["artifact_semantics_description"] = artifact_semantics_payload()["artifact_semantics_description"]
    checkpoint["dataset_artifact_semantics_version"] = str(dataset_summary["artifact_semantics_version"])
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "train_transitions": int(train_data["observations"].shape[0]),
        "val_transitions": int(val_data["observations"].shape[0]),
        "scalarizer": args.scalarizer,
        "alpha_conditioning_mode": args.alpha_conditioning_mode,
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "dataset_artifact_semantics_version": str(dataset_summary["artifact_semantics_version"]),
        **artifact_semantics_payload(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
