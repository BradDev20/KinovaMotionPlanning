from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .dataset import load_raw_records
from .run_layout import default_compare_output_dir
from .scalarization import hypervolume_2d, pareto_front
from .semantics import ARTIFACT_SEMANTICS_VERSION, semantics_version_from_payload
from .tasks import NONCONVEX_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare weighted-sum and weighted-max benchmark results.")
    parser.add_argument("--sum-dir", required=True, help="Directory containing weighted-sum datasets.")
    parser.add_argument("--max-dir", required=True, help="Directory containing weighted-max datasets.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for JSON/CSV comparison outputs.")
    parser.add_argument("--size-matched-samples", type=int, default=100, help="Bootstrap samples for size-matched max ablation.")
    parser.add_argument("--allow-profile-mismatch", action="store_true", help="Allow comparing datasets with different benchmark profile or geometry regime metadata.")
    return parser.parse_args()


def _coverage(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"pareto_count": 0, "hypervolume": 0.0}
    points = np.asarray([[record["length_cost"], record["obstacle_cost"]] for record in records], dtype=np.float64)
    front = pareto_front(points)
    reference = (
        float(points[:, 0].max() * 1.05 + 1e-6),
        float(points[:, 1].max() * 1.05 + 1e-6),
    )
    return {
        "pareto_count": int(front.shape[0]),
        "hypervolume": float(hypervolume_2d(points, reference)),
    }


def _bootstrap_ci(values: np.ndarray, repetitions: int = 2000, confidence: float = 0.95) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(repetitions):
        sampled = rng.choice(values, size=values.size, replace=True)
        samples.append(float(np.mean(sampled)))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _paired_significance(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "wilcoxon_stat": None, "wilcoxon_p": None}
    ci_low, ci_high = _bootstrap_ci(values)
    if values.size >= 2 and not np.allclose(values, values[0]):
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values)
        stat_value = float(wilcoxon_stat)
        p_value = float(wilcoxon_p)
    else:
        stat_value = None
        p_value = None
    return {
        "mean": float(np.mean(values)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_stat": stat_value,
        "wilcoxon_p": p_value,
    }


def _size_matched_summary(sum_records: list[dict], max_records: list[dict], sample_count: int) -> dict[str, float] | None:
    if len(max_records) <= len(sum_records) or not sum_records:
        return None
    rng = np.random.default_rng(0)
    sample_size = len(sum_records)
    hypervolumes = []
    for _ in range(sample_count):
        indices = rng.choice(len(max_records), size=sample_size, replace=False)
        sampled = [max_records[index] for index in indices]
        hypervolumes.append(_coverage(sampled)["hypervolume"])
    return {
        "sample_size": sample_size,
        "mean_hypervolume": float(np.mean(hypervolumes)),
        "std_hypervolume": float(np.std(hypervolumes)),
    }


def _dataset_dirs(root: Path) -> list[Path]:
    if (root / "dataset_summary.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("dataset_summary.json"))


def _load_dataset_info(root: str | Path) -> dict[tuple[str, str], dict]:
    datasets: dict[tuple[str, str], dict] = {}
    for dataset_dir in _dataset_dirs(Path(root)):
        summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
        family = str(summary.get("task_family", "all"))
        seed = f"seed_{int(summary.get('seed', 0)):04d}"
        datasets[(family, seed)] = {
            "dir": dataset_dir,
            "summary": summary,
            "records": load_raw_records(dataset_dir),
        }
    return datasets


def _metadata_signature(datasets: dict[tuple[str, str], dict]) -> set[tuple[str, str]]:
    return {
        (
            str(item["summary"].get("benchmark_profile", "baseline")),
            str(item["summary"].get("geometry_regime", "mixed")),
        )
        for item in datasets.values()
    }


def _artifact_semantics_signature(datasets: dict[tuple[str, str], dict]) -> set[str]:
    return {semantics_version_from_payload(item["summary"]) for item in datasets.values()}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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
    sum_datasets = _load_dataset_info(args.sum_dir)
    max_datasets = _load_dataset_info(args.max_dir)
    common_keys = sorted(sum_datasets.keys() & max_datasets.keys())
    if not common_keys:
        raise RuntimeError("No matching family/seed datasets found between sum and max outputs.")

    sum_signatures = _metadata_signature(sum_datasets)
    max_signatures = _metadata_signature(max_datasets)
    if not args.allow_profile_mismatch and (len(sum_signatures) != 1 or len(max_signatures) != 1 or sum_signatures != max_signatures):
        raise RuntimeError("Benchmark profile or geometry regime mismatch between sum and max datasets.")
    sum_semantics = _artifact_semantics_signature(sum_datasets)
    max_semantics = _artifact_semantics_signature(max_datasets)
    if len(sum_semantics) != 1 or len(max_semantics) != 1 or sum_semantics != max_semantics:
        raise RuntimeError("Artifact semantics mismatch between sum and max datasets.")
    semantics_version = next(iter(sum_semantics))
    if semantics_version != ARTIFACT_SEMANTICS_VERSION:
        raise RuntimeError(
            f"Benchmark comparison requires '{ARTIFACT_SEMANTICS_VERSION}' datasets, got '{semantics_version}'. "
            "Recollect datasets under the repaired full-body collision semantics before comparing."
        )

    output_dir = Path(args.output_dir) if args.output_dir else default_compare_output_dir(args.sum_dir, args.max_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = []
    family_summary: dict[str, dict[str, list[float]]] = {}
    regime_summary: dict[str, dict[str, list[float]]] = {}

    for family_name, seed_name in common_keys:
        sum_info = sum_datasets[(family_name, seed_name)]
        max_info = max_datasets[(family_name, seed_name)]
        sum_records = sum_info["records"]
        max_records = max_info["records"]
        sum_metrics = _coverage(sum_records)
        max_metrics = _coverage(max_records)
        geometry_regime = str(sum_info["summary"].get("geometry_regime", "mixed"))
        benchmark_profile = str(sum_info["summary"].get("benchmark_profile", "baseline"))
        nonconvex_win = int(
            family_name in NONCONVEX_FAMILIES
            and ((len(max_records) > len(sum_records)) or (max_metrics["hypervolume"] > sum_metrics["hypervolume"]))
        )
        row = {
            "family": family_name,
            "seed": seed_name,
            "benchmark_profile": benchmark_profile,
            "geometry_regime": geometry_regime,
            "sum_unique_count": len(sum_records),
            "max_unique_count": len(max_records),
            "unique_count_diff": len(max_records) - len(sum_records),
            "sum_nonconvex_route_count": int(sum_info["summary"].get("nonconvex_route_count", 0)),
            "max_nonconvex_route_count": int(max_info["summary"].get("nonconvex_route_count", 0)),
            "sum_hypervolume": sum_metrics["hypervolume"],
            "max_hypervolume": max_metrics["hypervolume"],
            "hypervolume_diff": max_metrics["hypervolume"] - sum_metrics["hypervolume"],
            "sum_pareto_count": sum_metrics["pareto_count"],
            "max_pareto_count": max_metrics["pareto_count"],
            "pareto_count_diff": max_metrics["pareto_count"] - sum_metrics["pareto_count"],
            "nonconvex_win_indicator": nonconvex_win,
            "size_matched_summary": _size_matched_summary(sum_records, max_records, args.size_matched_samples),
        }
        paired_rows.append(row)

        family_bucket = family_summary.setdefault(
            family_name,
            {"unique_count_diff": [], "hypervolume_diff": [], "pareto_count_diff": [], "wins": []},
        )
        family_bucket["unique_count_diff"].append(float(row["unique_count_diff"]))
        family_bucket["hypervolume_diff"].append(float(row["hypervolume_diff"]))
        family_bucket["pareto_count_diff"].append(float(row["pareto_count_diff"]))
        family_bucket["wins"].append(float(nonconvex_win))

        regime_bucket = regime_summary.setdefault(
            geometry_regime,
            {"unique_count_diff": [], "hypervolume_diff": [], "pareto_count_diff": [], "nonconvex_wins": []},
        )
        regime_bucket["unique_count_diff"].append(float(row["unique_count_diff"]))
        regime_bucket["hypervolume_diff"].append(float(row["hypervolume_diff"]))
        regime_bucket["pareto_count_diff"].append(float(row["pareto_count_diff"]))
        regime_bucket["nonconvex_wins"].append(float(nonconvex_win))

    unique_diffs = np.asarray([row["unique_count_diff"] for row in paired_rows], dtype=np.float64)
    hypervolume_diffs = np.asarray([row["hypervolume_diff"] for row in paired_rows], dtype=np.float64)
    pareto_diffs = np.asarray([row["pareto_count_diff"] for row in paired_rows], dtype=np.float64)

    family_report = {
        family_name: {
            "mean_unique_count_diff": _mean(values["unique_count_diff"]),
            "mean_hypervolume_diff": _mean(values["hypervolume_diff"]),
            "mean_pareto_count_diff": _mean(values["pareto_count_diff"]),
            "nonconvex_win_rate": _mean(values["wins"]),
        }
        for family_name, values in sorted(family_summary.items())
    }

    regime_report = {
        regime_name: {
            "mean_unique_count_diff": _mean(values["unique_count_diff"]),
            "mean_hypervolume_diff": _mean(values["hypervolume_diff"]),
            "mean_pareto_count_diff": _mean(values["pareto_count_diff"]),
            "nonconvex_win_rate": _mean(values["nonconvex_wins"]),
        }
        for regime_name, values in sorted(regime_summary.items())
    }

    nonconvex_rows = [row for row in paired_rows if row["family"] in NONCONVEX_FAMILIES]
    nonconvex_seed_wins: dict[str, int] = {}
    for row in nonconvex_rows:
        nonconvex_seed_wins.setdefault(row["seed"], 0)
        nonconvex_seed_wins[row["seed"]] = max(nonconvex_seed_wins[row["seed"]], int(row["nonconvex_win_indicator"]))

    report = {
        "seed_count": len({row["seed"] for row in paired_rows}),
        "family_seed_count": len(paired_rows),
        "metadata": {
            "sum_signatures": sorted(list(sum_signatures)),
            "max_signatures": sorted(list(max_signatures)),
        },
        "paired_rows": paired_rows,
        "unique_count_diff": _paired_significance(unique_diffs),
        "hypervolume_diff": _paired_significance(hypervolume_diffs),
        "pareto_count_diff": _paired_significance(pareto_diffs),
        "family_summary": family_report,
        "regime_summary": regime_report,
        "nonconvex_summary": {
            "row_count": len(nonconvex_rows),
            "winning_rows": int(sum(row["nonconvex_win_indicator"] for row in nonconvex_rows)),
            "seed_count": len(nonconvex_seed_wins),
            "winning_seed_count": int(sum(nonconvex_seed_wins.values())),
        },
    }

    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "paired_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in paired_rows[0].keys() if key != "size_matched_summary"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in paired_rows:
            writer.writerow({key: value for key, value in row.items() if key != "size_matched_summary"})

    print(f"Saved benchmark report to {output_dir}")


if __name__ == "__main__":
    main()
