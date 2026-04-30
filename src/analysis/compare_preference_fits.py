from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare preference fit results across two policies.")
    parser.add_argument("--fits", nargs=2, required=True, metavar="PATH", help="Paths to two preference_fit.json files.")
    parser.add_argument("--labels", nargs=2, default=None, metavar="LABEL", help="Display labels for each fit (default: inferred from path).")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    return parser.parse_args()


def _infer_label(path: Path) -> str:
    for part in reversed(path.parts):
        name = part.strip().lower()
        if name in {"max", "sum"}:
            return "WM" if name == "max" else "WS"
    return path.parent.name


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Okabe-Ito colorblind-safe palette
_COLORS = ["#0072B2", "#D55E00"]  # blue (max), vermillion (sum)
_CORRECT_ALPHA = 1.0
_WRONG_ALPHA = 0.45


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required.") from exc

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    })

    paths = [Path(p) for p in args.fits]
    labels = args.labels if args.labels else [_infer_label(p) for p in paths]
    reports = [_load(p) for p in paths]

    correct_nll = []
    wrong_nll = []
    for report in reports:
        true_model = report["synthetic"]["true_model"]
        wrong_model = "sum" if true_model == "max" else "max"
        correct_nll.append(report["metrics"][true_model]["test"]["nll"])
        wrong_nll.append(report["metrics"][wrong_model]["test"]["nll"])

    gaps = [w - c for c, w in zip(correct_nll, wrong_nll)]
    x = np.arange(len(labels))
    width = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.subplots_adjust(wspace=0.38)

    # --- Left: correct vs wrong model NLL per policy ---
    ax = axes[0]
    for i, (c_nll, w_nll, color, label) in enumerate(zip(correct_nll, wrong_nll, _COLORS, labels)):
        ax.bar(i - width / 2, c_nll, width, color=color, alpha=_CORRECT_ALPHA, label=f"{label}: correct model")
        ax.bar(i + width / 2, w_nll, width, color=color, alpha=_WRONG_ALPHA, label=f"{label}: wrong model")

    top_left = max(wrong_nll) * 1.28
    ax.set_ylim(0, top_left)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test NLL per pair")
    ax.set_title("Model NLL by Policy")

    for i, (c_nll, w_nll) in enumerate(zip(correct_nll, wrong_nll)):
        offset = top_left * 0.025
        ax.text(i - width / 2, c_nll + offset, f"{c_nll:.3f}", ha="center", va="bottom", fontsize=10)
        ax.text(i + width / 2, w_nll + offset, f"{w_nll:.3f}", ha="center", va="bottom", fontsize=10)

    legend_handles = []
    for i, (color, label) in enumerate(zip(_COLORS, labels)):
        legend_handles.append(mpatches.Patch(facecolor=color, alpha=_CORRECT_ALPHA, label=f"{label}: correct"))
        legend_handles.append(mpatches.Patch(facecolor=color, alpha=_WRONG_ALPHA, label=f"{label}: wrong"))
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right", framealpha=0.9)

    # --- Right: NLL gap per policy ---
    ax2 = axes[1]
    bars = ax2.bar(x, gaps, width * 1.9, color=_COLORS, alpha=0.9)
    gap_top = max(gaps) * 1.3
    ax2.set_ylim(0, gap_top)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("NLL Gap")
    ax2.set_title("Preference Discrimination Gap")
    for bar, gap in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2, gap + gap_top * 0.025,
                 f"{gap:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    legend_handles2 = [
        mpatches.Patch(facecolor=_COLORS[i], alpha=0.9, label=labels[i]) for i in range(len(labels))
    ]
    ax2.legend(handles=legend_handles2, fontsize=9, loc="upper right", framealpha=0.9)

    fig.suptitle("Preference Model Discrimination: Weighted Max vs. Weighted Sum Policy",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
