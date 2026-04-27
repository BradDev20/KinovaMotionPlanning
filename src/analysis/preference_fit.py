from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


PreferenceModel = Literal["sum", "max"]


@dataclasses.dataclass(frozen=True)
class RolloutPoint:
    rollout_index: int
    task_id: str
    family: str
    source: str
    scalarizer: str
    alpha: float | None
    costs: tuple[float, float]  # (length_cost, obstacle_cost)
    success: bool


@dataclasses.dataclass(frozen=True)
class PairwisePreference:
    task_id: str
    family: str
    a: RolloutPoint
    b: RolloutPoint
    y: int  # 1 => prefer a, 0 => prefer b


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _extract_costs(item: dict[str, Any]) -> tuple[float, float] | None:
    if "objective_total" in item and item["objective_total"] is not None:
        vec = item["objective_total"]
        if isinstance(vec, (list, tuple, np.ndarray)) and len(vec) >= 2:
            return (float(vec[0]), float(vec[1]))
    if "objective_vector" in item and item["objective_vector"] is not None:
        vec = item["objective_vector"]
        if isinstance(vec, (list, tuple, np.ndarray)) and len(vec) >= 2:
            # Step objective, not total. Still usable if you only have per-step points.
            return (float(vec[0]), float(vec[1]))
    if "length_cost" in item and "obstacle_cost" in item:
        return (float(item["length_cost"]), float(item["obstacle_cost"]))
    return None


def load_rollouts(paths: Iterable[str | Path]) -> list[RolloutPoint]:
    points: list[RolloutPoint] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise RuntimeError(f"Unsupported rollouts format in {path}: expected list, got {type(data)}")
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or (item.get("task_spec") or {}).get("task_id") or "")
            if not task_id:
                continue
            task_spec = item.get("task_spec") if isinstance(item.get("task_spec"), dict) else {}
            family = str(task_spec.get("family", "base"))
            source = str(item.get("source", "unknown"))
            scalarizer = str(item.get("scalarizer", "unknown"))
            alpha = item.get("alpha", None)
            try:
                alpha_f = float(alpha) if alpha is not None else None
            except (TypeError, ValueError):
                alpha_f = None
            costs = _extract_costs(item)
            if costs is None:
                continue
            success = bool(item.get("success", True))
            points.append(
                RolloutPoint(
                    rollout_index=int(idx),
                    task_id=task_id,
                    family=family,
                    source=source,
                    scalarizer=scalarizer,
                    alpha=alpha_f,
                    costs=(float(costs[0]), float(costs[1])),
                    success=success,
                )
            )
    return points


def filter_points(
    points: list[RolloutPoint],
    *,
    family: str | None,
    source: str | None,
    scalarizer: str | None,
    successes_only: bool,
) -> list[RolloutPoint]:
    out: list[RolloutPoint] = []
    for p in points:
        if successes_only and not p.success:
            continue
        if family and p.family != family:
            continue
        if source and p.source != source:
            continue
        if scalarizer and p.scalarizer != scalarizer:
            continue
        out.append(p)
    return out


def sample_pairs(points: list[RolloutPoint], *, pairs_per_task: int, seed: int) -> list[tuple[RolloutPoint, RolloutPoint]]:
    by_task: dict[str, list[RolloutPoint]] = defaultdict(list)
    for p in points:
        by_task[p.task_id].append(p)

    rng = np.random.default_rng(int(seed))
    pairs: list[tuple[RolloutPoint, RolloutPoint]] = []
    for task_id, items in sorted(by_task.items()):
        if len(items) < 2:
            continue
        # Sample without trying to be fancy; repeated pairs are fine for small datasets.
        count = int(pairs_per_task)
        for _ in range(count):
            a_idx, b_idx = rng.integers(0, len(items), size=2)
            if a_idx == b_idx:
                continue
            a = items[int(a_idx)]
            b = items[int(b_idx)]
            pairs.append((a, b))
    rng.shuffle(pairs)
    return pairs


def _cost_sum(costs: tuple[float, float], w_length: float) -> float:
    w = float(np.clip(w_length, 0.0, 1.0))
    return w * float(costs[0]) + (1.0 - w) * float(costs[1])


def _cost_max(costs: tuple[float, float], w_length: float) -> float:
    w = float(np.clip(w_length, 0.0, 1.0))
    return max(w * float(costs[0]), (1.0 - w) * float(costs[1]))


def _pair_prob_prefer_a(
    *,
    model: PreferenceModel,
    a_costs: tuple[float, float],
    b_costs: tuple[float, float],
    w_length: float,
    temperature: float,
) -> float:
    temp = max(float(temperature), 1e-6)
    if model == "sum":
        c_a = _cost_sum(a_costs, w_length=w_length)
        c_b = _cost_sum(b_costs, w_length=w_length)
    elif model == "max":
        c_a = _cost_max(a_costs, w_length=w_length)
        c_b = _cost_max(b_costs, w_length=w_length)
    else:
        raise ValueError(f"Unknown model: {model}")
    # Lower cost should be preferred, so p(prefer a) increases as (c_b - c_a) increases.
    return float(np.clip(_sigmoid((c_b - c_a) / temp), 1e-9, 1.0 - 1e-9))


def make_synthetic_preferences(
    pairs: list[tuple[RolloutPoint, RolloutPoint]],
    *,
    true_model: PreferenceModel,
    w_true: float,
    temperature: float,
    seed: int,
) -> list[PairwisePreference]:
    rng = np.random.default_rng(int(seed))
    prefs: list[PairwisePreference] = []
    for a, b in pairs:
        p = _pair_prob_prefer_a(
            model=true_model,
            a_costs=a.costs,
            b_costs=b.costs,
            w_length=w_true,
            temperature=temperature,
        )
        y = int(rng.random() < p)
        prefs.append(PairwisePreference(task_id=a.task_id, family=a.family, a=a, b=b, y=y))
    return prefs


def fit_preference_model(
    prefs: list[PairwisePreference],
    *,
    model: PreferenceModel,
    w_grid: np.ndarray,
    temperature_grid: np.ndarray,
) -> dict[str, float]:
    if not prefs:
        raise RuntimeError("No preferences provided.")

    best = {"w_length": float(w_grid[0]), "temperature": float(temperature_grid[0]), "nll": float("inf")}
    for temp in temperature_grid:
        for w in w_grid:
            nll = 0.0
            for pref in prefs:
                p = _pair_prob_prefer_a(
                    model=model,
                    a_costs=pref.a.costs,
                    b_costs=pref.b.costs,
                    w_length=float(w),
                    temperature=float(temp),
                )
                nll += -math.log(p) if pref.y == 1 else -math.log(1.0 - p)
            if nll < best["nll"]:
                best = {"w_length": float(w), "temperature": float(temp), "nll": float(nll)}
    return best


def eval_preference_model(
    prefs: list[PairwisePreference],
    *,
    model: PreferenceModel,
    w_length: float,
    temperature: float,
) -> dict[str, float]:
    if not prefs:
        return {"count": 0.0, "acc": 0.0, "nll": 0.0}
    correct = 0
    nll = 0.0
    for pref in prefs:
        p = _pair_prob_prefer_a(
            model=model,
            a_costs=pref.a.costs,
            b_costs=pref.b.costs,
            w_length=w_length,
            temperature=temperature,
        )
        yhat = 1 if p >= 0.5 else 0
        correct += int(yhat == pref.y)
        nll += -math.log(p) if pref.y == 1 else -math.log(1.0 - p)
    return {"count": float(len(prefs)), "acc": float(correct) / float(len(prefs)), "nll": float(nll) / float(len(prefs))}


def _split_prefs(prefs: list[PairwisePreference], *, train_frac: float, seed: int) -> tuple[list[PairwisePreference], list[PairwisePreference]]:
    rng = np.random.default_rng(int(seed))
    idxs = np.arange(len(prefs))
    rng.shuffle(idxs)
    cut = int(round(float(train_frac) * float(len(prefs))))
    cut = max(1, min(cut, len(prefs) - 1)) if len(prefs) >= 2 else len(prefs)
    train = [prefs[int(i)] for i in idxs[:cut]]
    test = [prefs[int(i)] for i in idxs[cut:]]
    return train, test


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit weighted-sum vs weighted-max preference models from rollouts.pkl.")
    parser.add_argument(
        "--rollouts",
        nargs="+",
        required=True,
        help="One or more rollouts.pkl paths (planner and/or offline_rl).",
    )
    parser.add_argument("--family", default=None, help="Optional task family filter (e.g., double_corridor).")
    parser.add_argument("--source", default=None, help="Optional source filter (planner or offline_rl).")
    parser.add_argument("--scalarizer", default=None, help="Optional scalarizer filter (sum or max).")
    parser.add_argument("--successes-only", action="store_true", help="Only use successful rollouts.")
    parser.add_argument("--pairs-per-task", type=int, default=50, help="Number of rollout pairs to sample per task.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--prefs", choices=["synthetic"], default="synthetic", help="Preference label source.")
    parser.add_argument("--true-model", choices=["sum", "max"], default="max", help="Synthetic preference generator.")
    parser.add_argument("--w-true", type=float, default=0.5, help="True length weight for synthetic preferences (in [0,1]).")
    parser.add_argument("--temp-true", type=float, default=0.05, help="Preference noise temperature for synthetic prefs.")

    parser.add_argument("--train-frac", type=float, default=0.8, help="Train/test split fraction.")
    parser.add_argument("--w-grid-size", type=int, default=201, help="Grid size for w in [0,1].")
    parser.add_argument(
        "--temp-grid",
        default="0.01,0.02,0.05,0.1,0.2",
        help="Comma-separated temperature grid for fitting.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--plot", default=None, help="Optional PNG output path (summary bars for test metrics).")
    args = parser.parse_args(argv)

    points = load_rollouts(args.rollouts)
    points = filter_points(
        points,
        family=args.family,
        source=args.source,
        scalarizer=args.scalarizer,
        successes_only=bool(args.successes_only),
    )
    if not points:
        raise RuntimeError("No rollout points matched the provided filters.")

    pairs = sample_pairs(points, pairs_per_task=int(args.pairs_per_task), seed=int(args.seed))
    if not pairs:
        raise RuntimeError("No task had at least two rollouts after filtering; cannot sample pairs.")

    prefs = make_synthetic_preferences(
        pairs,
        true_model=str(args.true_model),
        w_true=float(args.w_true),
        temperature=float(args.temp_true),
        seed=int(args.seed),
    )

    train, test = _split_prefs(prefs, train_frac=float(args.train_frac), seed=int(args.seed))
    w_grid = np.linspace(0.0, 1.0, int(args.w_grid_size))
    temperature_grid = np.asarray([float(x.strip()) for x in str(args.temp_grid).split(",") if x.strip()], dtype=np.float64)
    if temperature_grid.size == 0:
        temperature_grid = np.asarray([0.05], dtype=np.float64)

    fit_sum = fit_preference_model(train, model="sum", w_grid=w_grid, temperature_grid=temperature_grid)
    fit_max = fit_preference_model(train, model="max", w_grid=w_grid, temperature_grid=temperature_grid)

    sum_train = eval_preference_model(train, model="sum", w_length=fit_sum["w_length"], temperature=fit_sum["temperature"])
    sum_test = eval_preference_model(test, model="sum", w_length=fit_sum["w_length"], temperature=fit_sum["temperature"])
    max_train = eval_preference_model(train, model="max", w_length=fit_max["w_length"], temperature=fit_max["temperature"])
    max_test = eval_preference_model(test, model="max", w_length=fit_max["w_length"], temperature=fit_max["temperature"])

    report = {
        "filters": {
            "family": args.family,
            "source": args.source,
            "scalarizer": args.scalarizer,
            "successes_only": bool(args.successes_only),
        },
        "data": {
            "rollout_points": int(len(points)),
            "pairs": int(len(pairs)),
            "prefs": int(len(prefs)),
            "train": int(len(train)),
            "test": int(len(test)),
        },
        "synthetic": {
            "true_model": str(args.true_model),
            "w_true": float(args.w_true),
            "temperature_true": float(args.temp_true),
        },
        "fit": {
            "sum": fit_sum,
            "max": fit_max,
        },
        "metrics": {
            "sum": {"train": sum_train, "test": sum_test},
            "max": {"train": max_train, "test": max_test},
        },
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        out_path = Path(args.output)
        _ensure_parent(out_path)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("matplotlib is required for --plot output.") from exc

        plot_path = Path(args.plot)
        _ensure_parent(plot_path)

        labels = ["weighted_sum", "weighted_max"]
        test_acc = [report["metrics"]["sum"]["test"]["acc"], report["metrics"]["max"]["test"]["acc"]]
        test_nll = [report["metrics"]["sum"]["test"]["nll"], report["metrics"]["max"]["test"]["nll"]]

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=160)
        axes[0].bar(labels, test_acc, color=["#4C78A8", "#F58518"])
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Preference Prediction Accuracy (Test)")
        axes[0].set_ylabel("Accuracy")
        axes[0].tick_params(axis="x", rotation=15)

        axes[1].bar(labels, test_nll, color=["#4C78A8", "#F58518"])
        axes[1].set_title("Preference NLL (Test)")
        axes[1].set_ylabel("NLL (lower is better)")
        axes[1].tick_params(axis="x", rotation=15)

        title_bits = []
        if args.family:
            title_bits.append(f"family={args.family}")
        if args.source:
            title_bits.append(f"source={args.source}")
        if args.scalarizer:
            title_bits.append(f"scalarizer={args.scalarizer}")
        if title_bits:
            fig.suptitle(", ".join(title_bits), fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
