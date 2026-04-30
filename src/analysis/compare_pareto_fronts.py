from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Side-by-side Pareto front comparison for two policies.")
    parser.add_argument("--evaluation-dirs", nargs=2, required=True, metavar="DIR", help="Evaluation directories containing metrics.csv (max first, then sum).")
    parser.add_argument("--labels", nargs=2, default=None, metavar="LABEL", help="Panel labels (default: inferred from directory name).")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    return parser.parse_args()


def _infer_label(path: Path) -> str:
    for part in reversed(path.parts):
        name = part.strip().lower()
        if name in {"max", "sum"}:
            return f"Weighted {'Max' if name == 'max' else 'Sum'} Policy"
    return path.parent.name


def _load_successes(evaluation_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = evaluation_dir / "metrics.csv"
    costs, alphas = [], []
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("success", "False") != "True":
                continue
            costs.append([float(row["length_objective"]), float(row["obstacle_objective"])])
            alphas.append(float(row["alpha"]))
    return np.asarray(costs, dtype=np.float64), np.asarray(alphas, dtype=np.float64)


def _pareto_front_sorted(points: np.ndarray) -> np.ndarray:
    from src.morl.scalarization import pareto_front
    front = pareto_front(points)
    return front[front[:, 0].argsort()]


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colorbar
    except ImportError as exc:
        raise RuntimeError("matplotlib is required.") from exc

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    paths = [Path(d) for d in args.evaluation_dirs]
    labels = args.labels if args.labels else [_infer_label(p) for p in paths]
    panel_colors = ["#0072B2", "#D55E00"]

    all_costs = [_load_successes(p) for p in paths]

    # Shared axis limits across both panels
    all_length = np.concatenate([c[0][:, 0] for c in all_costs if c[0].size])
    all_obstacle = np.concatenate([c[0][:, 1] for c in all_costs if c[0].size])
    pad_x = (all_length.max() - all_length.min()) * 0.08
    pad_y = (all_obstacle.max() - all_obstacle.min()) * 0.08
    xlim = (max(all_length.min() - pad_x, 0), all_length.max() + pad_x)
    ylim = (max(all_obstacle.min() - pad_y, 0), all_obstacle.max() + pad_y)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200, sharey=True)
    fig.subplots_adjust(wspace=0.12)

    cmap = cm.plasma
    scatter_ref = None

    for idx, (ax, (costs, alphas), label, color) in enumerate(zip(axes, all_costs, labels, panel_colors)):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Length Cost")
        if idx == 0:
            ax.set_ylabel("Obstacle Cost")
        ax.set_title(f"{label}", fontweight="semibold")

        if costs.size == 0:
            ax.text(0.5, 0.5, "No successful rollouts", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            continue

        sc = ax.scatter(costs[:, 0], costs[:, 1], c=alphas, cmap=cmap,
                        vmin=0.0, vmax=1.0, s=55, edgecolors=color,
                        linewidths=0.6, zorder=3, alpha=0.9)
        scatter_ref = sc

        front = _pareto_front_sorted(costs)
        if front.size:
            ax.plot(front[:, 0], front[:, 1], color=color, linewidth=2.0,
                    linestyle="-", zorder=4, label="Pareto Front")
            ax.scatter(front[:, 0], front[:, 1], s=80, color=color,
                       edgecolors="white", linewidths=1.0, zorder=5)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

        n_success = costs.shape[0]
        n_front = front.shape[0] if front.size else 0
        ax.text(0.03, 0.97, f"n={n_success}  front={n_front}",
                transform=ax.transAxes, va="top", fontsize=10, color="#333333")

    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=axes.tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Alpha", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Pareto Front: Successful Rollouts by Policy",
                 fontsize=14, fontweight="bold", y=1.02)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto comparison to {output_path}")


if __name__ == "__main__":
    main()
