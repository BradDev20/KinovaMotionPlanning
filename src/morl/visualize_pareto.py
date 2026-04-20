from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .dataset import load_raw_records
from .scalarization import hypervolume_2d, pareto_front
from .tasks import NONCONVEX_FAMILIES

POINT_SIZE = 70
FRONT_POINT_SIZE = 120
FRONT_LINE_WIDTH = 3.0
SCATTER_EDGE_WIDTH = 0.6


def _pretty_family_name(name: str) -> str:
    mapping = {
        "offset_gate": "Offset Gate",
        "pinch_bottleneck": "Pinch Bottleneck",
        "double_corridor": "Double Corridor",
        "culdesac_escape": "Cul-de-sac Escape",
        "corridor_left_right": "Left/Right Corridor",
        "pinch_point": "Pinch Point",
        "stacked_detour": "Stacked Detour",
        "asymmetric_safe_margin": "Asymmetric Safe Margin",
        "evaluation": "Evaluation",
        "all": "All Families",
    }
    normalized = str(name).strip().lower()
    if normalized in mapping:
        return mapping[normalized]
    return str(name).replace("_", " ").title()


def _pretty_series_name(planner_mode: str, task_family: str) -> str:
    mode_name = {
        "sum": "Weighted Sum",
        "max": "Weighted Max",
    }.get(str(planner_mode).strip().lower(), str(planner_mode).replace("_", " ").title())
    return f"{mode_name}: {_pretty_family_name(task_family)}"


def _pretty_mode_name(planner_mode: str) -> str:
    return {
        "sum": "Weighted Sum",
        "max": "Weighted Max",
    }.get(str(planner_mode).strip().lower(), str(planner_mode).replace("_", " ").title())


def _load_planner_summary(path: Path) -> dict | None:
    candidates = [path / "dataset_summary.json", path.parent / "dataset_summary.json", path.parent.parent / "dataset_summary.json"]
    for summary_path in candidates:
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def _infer_planner_mode(path: Path) -> str:
    for candidate in (path, *path.parents):
        name = candidate.name.strip().lower()
        if name in {"sum", "max"}:
            return name
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize planner and RL Pareto fronts.")
    parser.add_argument("--planner-dirs", nargs="*", default=[], help="Planner dataset directories from collect_dataset.")
    parser.add_argument("--evaluation-dirs", nargs="*", default=[], help="Evaluation directories containing metrics.csv.")
    parser.add_argument("--source", choices=["both", "planner", "offline_rl"], default="both", help="Select whether to show planner data, offline RL data, or both.")
    parser.add_argument("--output", type=str, default="data/morl/pareto_comparison.png", help="Output PNG path.")
    parser.add_argument("--summary-output", type=str, default=None, help="Optional JSON summary output.")
    parser.add_argument("--group-by-family", action="store_true", help="Create small multiples by task family for planner datasets.")
    parser.add_argument("--coverage-only", action="store_true", help="Plot Pareto-front points only instead of all trajectories.")
    parser.add_argument("--color-by-alpha", action="store_true", help="Color planner points by alpha instead of series color.")
    parser.add_argument("--nonconvex-only", action="store_true", help="Plot only non-convex families.")
    return parser.parse_args()


def evaluation_records(evaluation_dir: str | Path) -> list[dict]:
    csv_path = Path(evaluation_dir) / "metrics.csv"
    points = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                {
                    "length_cost": float(row["length_objective"]),
                    "obstacle_cost": float(row["obstacle_objective"]),
                    "alpha": float(row["alpha"]),
                    "family": "evaluation",
                    "benchmark_profile": row.get("benchmark_profile", "unknown"),
                    "geometry_regime": row.get("geometry_regime", "evaluation"),
                }
            )
    return points


def _planner_dataset_dirs(root: Path) -> list[Path]:
    if (root / "dataset_summary.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("dataset_summary.json"))


def series_label(path_like: str | Path, source_type: str) -> str:
    path = Path(path_like)
    if source_type == "planner":
        summary = _load_planner_summary(path)
        if summary is not None:
            planner_mode = str(summary.get("planner_mode", "")).strip()
            task_family = str(summary.get("task_family", path.name)).strip()
            if planner_mode:
                return _pretty_mode_name(planner_mode)
        planner_mode = _infer_planner_mode(path)
        if planner_mode:
            return _pretty_mode_name(planner_mode)
        return _pretty_family_name(path.name)
    if source_type == "offline_rl":
        if path.name == "evaluation" and path.parent.name in {"sum", "max"}:
            return f"IQL Eval: {_pretty_mode_name(path.parent.name)}"
        if path.name == "evaluation" and path.parent.name:
            return f"IQL Eval: {path.parent.name}"
    if path.name == "evaluation" and path.parent.name:
        return path.parent.name
    return path.name


def series_style(label: str, source_type: str, index: int) -> tuple[str, str, str]:
    normalized = label.lower()
    if "sum" in normalized and "iql" not in normalized:
        return "#1f77b4", "o", "-"
    if "max" in normalized and "iql" not in normalized:
        return "#d62728", "o", "-"
    if "sum" in normalized and "iql" in normalized:
        return "#2ca02c", "s", "--"
    if "max" in normalized and "iql" in normalized:
        return "#8b0000", "s", "--"
    fallback_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    color = fallback_colors[index % len(fallback_colors)]
    marker = "o" if source_type == "planner" else "s"
    linestyle = "-" if source_type == "planner" else "--"
    return color, marker, linestyle


def _points(records: list[dict], coverage_only: bool) -> np.ndarray:
    points = np.asarray([[record["length_cost"], record["obstacle_cost"]] for record in records], dtype=np.float64)
    return pareto_front(points) if coverage_only else points


def _summary(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"count": 0, "pareto_count": 0, "hypervolume": 0.0}
    points = np.asarray([[record["length_cost"], record["obstacle_cost"]] for record in records], dtype=np.float64)
    front = pareto_front(points)
    reference = (
        float(points[:, 0].max() * 1.05 + 1e-6),
        float(points[:, 1].max() * 1.05 + 1e-6),
    )
    return {
        "count": int(points.shape[0]),
        "pareto_count": int(front.shape[0]),
        "hypervolume": float(hypervolume_2d(points, reference)),
    }


def _group_records(records: list[dict], group_by_family: bool, nonconvex_only: bool) -> dict[str, list[dict]]:
    filtered = [
        record
        for record in records
        if not nonconvex_only or str(record.get("family", record.get("task_spec", {}).get("family", "unknown"))) in NONCONVEX_FAMILIES
    ]
    if not group_by_family:
        return {"all": filtered}
    grouped: dict[str, list[dict]] = {}
    for record in filtered:
        grouped.setdefault(str(record.get("family", record.get("task_spec", {}).get("family", "unknown"))), []).append(record)
    return grouped


def _legend_label(label: str, count: int, pareto_count: int) -> str:
    if count <= 0:
        return f"{label} (n=0)"
    return f"{label} (n={count}, front={pareto_count})"


def _style_axis(axis) -> None:
    axis.set_xlabel("Length Objective", fontsize=18, fontweight="semibold")
    axis.set_ylabel("Obstacle Objective", fontsize=18, fontweight="semibold")
    axis.grid(True, alpha=0.18, linewidth=0.8)
    axis.tick_params(labelsize=15, width=1.0)


def _plot_series(axis, records: list[dict], label: str, source_type: str, index: int, coverage_only: bool, color_by_alpha: bool) -> dict[str, object]:
    color, marker, linestyle = series_style(label, source_type, index)
    summary = _summary(records)
    summary["label"] = label
    summary["source"] = source_type
    summary["color"] = color
    legend_label = _legend_label(label, count=int(summary["count"]), pareto_count=int(summary["pareto_count"]))
    points = _points(records, coverage_only=coverage_only)
    if points.size == 0:
        axis.scatter([], [], s=POINT_SIZE, c=color, marker=marker, edgecolors="black", linewidths=SCATTER_EDGE_WIDTH, label=legend_label)
        return summary

    if source_type == "planner" and color_by_alpha and not coverage_only:
        alphas = np.asarray([float(record["alpha"]) for record in records], dtype=np.float64)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=POINT_SIZE,
            alpha=1.0,
            c=alphas,
            cmap="viridis",
            marker=marker,
            edgecolors=color,
            linewidths=SCATTER_EDGE_WIDTH,
            label=legend_label,
            zorder=2,
        )
    else:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=POINT_SIZE,
            alpha=1.0,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=SCATTER_EDGE_WIDTH,
            label=legend_label,
            zorder=2,
        )

    front = pareto_front(np.asarray([[record["length_cost"], record["obstacle_cost"]] for record in records], dtype=np.float64))
    if front.size:
        axis.plot(front[:, 0], front[:, 1], color=color, linewidth=FRONT_LINE_WIDTH, linestyle=linestyle, zorder=3)
        axis.scatter(
            front[:, 0],
            front[:, 1],
            s=FRONT_POINT_SIZE,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
        )
    return summary


def _metadata_for_planner_dirs(paths: list[Path]) -> list[dict[str, str]]:
    metadata = []
    for path in paths:
        summary = _load_planner_summary(path)
        if summary is None:
            continue
        metadata.append(
            {
                "benchmark_profile": str(summary.get("benchmark_profile", "baseline")),
                "geometry_regime": str(summary.get("geometry_regime", "mixed")),
                "task_family": str(summary.get("task_family", "all")),
            }
        )
    return metadata


def _figure_title(source: str) -> str:
    if source == "planner":
        return "Planner Pareto Comparison"
    if source == "offline_rl":
        return "Offline RL Pareto Comparison"
    return "Planner vs Offline RL Pareto Comparison"


def _title_with_family(base_title: str, title_suffix: str, task_families: list[str]) -> str:
    visible_families = [family for family in task_families if family and family != "all"]
    if len(visible_families) == 1:
        return f"{base_title}: {_pretty_family_name(visible_families[0])}\n{title_suffix}"
    if len(visible_families) > 1:
        pretty_families = ", ".join(_pretty_family_name(family) for family in visible_families)
        return f"{base_title}: {pretty_families}\n{title_suffix}"
    return f"{base_title}\n{title_suffix}"


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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting depends on matplotlib
        raise RuntimeError("matplotlib is required for Pareto visualization.") from exc

    planner_series = []
    planner_dirs: list[Path] = []
    if args.source in {"both", "planner"}:
        for dataset_root in args.planner_dirs:
            for dataset_dir in _planner_dataset_dirs(Path(dataset_root)):
                records = load_raw_records(dataset_dir)
                planner_dirs.append(dataset_dir)
                planner_series.append(("planner", series_label(dataset_dir, "planner"), records))
    evaluation_series = []
    if args.source in {"both", "offline_rl"}:
        for evaluation_dir in args.evaluation_dirs:
            records = evaluation_records(evaluation_dir)
            evaluation_series.append(("offline_rl", series_label(evaluation_dir, "offline_rl"), records))

    series = planner_series + evaluation_series
    if not series:
        raise RuntimeError("No Pareto data found. Provide planner and/or evaluation directories.")

    planner_metadata = _metadata_for_planner_dirs(planner_dirs)
    profiles = sorted({item["benchmark_profile"] for item in planner_metadata}) or ["unknown"]
    regimes = sorted({item["geometry_regime"] for item in planner_metadata}) or ["unknown"]
    task_families = sorted({item["task_family"] for item in planner_metadata}) or ["all"]
    title_suffix = f"Profiles: {', '.join(profiles)} | Regimes: {', '.join(regimes)}"
    if args.nonconvex_only:
        title_suffix += " | Non-Convex Only"

    if args.group_by_family and planner_series:
        family_names = sorted(
            {
                family
                for _, _, records in planner_series
                for family in _group_records(records, True, args.nonconvex_only).keys()
            }
        )
        columns = min(2, max(1, len(family_names)))
        rows = int(np.ceil(len(family_names) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(8 * columns, 6 * rows), squeeze=False)
        summary = {"series": [], "group_by_family": True, "metadata": {"profiles": profiles, "regimes": regimes}}
        for axis in axes.reshape(-1):
            axis.set_visible(False)
        for family_index, family_name in enumerate(family_names):
            axis = axes.reshape(-1)[family_index]
            axis.set_visible(True)
            for series_index, (source_type, label, records) in enumerate(planner_series):
                family_records = _group_records(records, True, args.nonconvex_only).get(family_name, [])
                summary["series"].append(
                    {
                        "family": family_name,
                        **_plot_series(
                            axis,
                            family_records,
                            label=label,
                            source_type=source_type,
                            index=series_index,
                            coverage_only=args.coverage_only,
                            color_by_alpha=args.color_by_alpha,
                        ),
                    }
                )
            axis.set_title(f"Family: {_pretty_family_name(family_name)}")
            _style_axis(axis)
            axis.legend(loc="best", fontsize=11, framealpha=0.95)
        figure.suptitle(
            _title_with_family(_figure_title(args.source), title_suffix, task_families),
            fontsize=22,
            fontweight="semibold",
        )
    else:
        figure, axis = plt.subplots(figsize=(11, 8))
        summary = {"series": [], "group_by_family": False, "metadata": {"profiles": profiles, "regimes": regimes}}
        for index, (source_type, label, records) in enumerate(series):
            filtered_records = _group_records(records, False, args.nonconvex_only)["all"]
            summary["series"].append(
                _plot_series(
                    axis,
                    filtered_records,
                    label=label,
                    source_type=source_type,
                    index=index,
                    coverage_only=args.coverage_only,
                    color_by_alpha=args.color_by_alpha,
                )
            )
        _style_axis(axis)
        axis.set_title(
            _title_with_family(_figure_title(args.source), title_suffix, task_families),
            fontsize=22,
            fontweight="semibold",
        )
        axis.legend(loc="best", fontsize=14, framealpha=0.95)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_path, dpi=240)

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Pareto plot to {output_path}")


if __name__ == "__main__":
    main()
