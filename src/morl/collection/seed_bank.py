"""
Seed bank management for the trajectory collection process.
We keep a "bank" of good trajectories to help the planner start 
with a decent guess instead of starting from scratch every time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset import trajectory_distance

MAX_SEEDS_PER_FAMILY = 12
SEED_NOISE_SCALE = 0.03
SEED_DISTINCT_TOL = 0.20

@dataclass(frozen=True)
class SeedEntry:
    """
    A single saved trajectory that we can use to warm-start the planner.
    Includes the actual points and some cost info.
    """
    trajectory: np.ndarray  # float32, shape=(N, 7)
    start_config: np.ndarray  # float32, shape=(7,)
    goal_config: np.ndarray  # float32, shape=(7,)
    task_id: str = ""
    alpha: float | None = None
    length_cost: float | None = None
    obstacle_cost: float | None = None

def _select_family_seed(
    seeds: list[SeedEntry],
    *,
    start_config: np.ndarray,
    goal_config: np.ndarray,
    rng: np.random.Generator,
    top_k: int = 3,
) -> SeedEntry | None:
    """
    Quick helper to grab a decent seed from the family bank. 
    It picks from the top_k seeds that have the closest start/goal configs.
    """
    if not seeds:
        return None
    scored: list[tuple[float, int]] = []
    for idx, entry in enumerate(seeds):
        mismatch = float(np.linalg.norm(start_config - entry.start_config) + np.linalg.norm(goal_config - entry.goal_config))
        scored.append((mismatch, idx))
    scored.sort(key=lambda item: item[0])
    k = max(1, min(int(top_k), len(scored)))
    candidates = [idx for _, idx in scored[:k]]
    chosen = int(rng.choice(np.asarray(candidates, dtype=np.int32)))
    return seeds[chosen]

def _select_extreme_seed(
    seeds: list[SeedEntry],
    *,
    kind: str,
) -> SeedEntry | None:
    """
    Find either the safest or the riskiest (shortest) seed in the list.
    """
    if not seeds:
        return None
    if kind == "safe":
        candidates = [seed for seed in seeds if seed.obstacle_cost is not None]
        return min(candidates, key=lambda seed: float(seed.obstacle_cost)) if candidates else None
    if kind == "risky":
        candidates = [seed for seed in seeds if seed.length_cost is not None]
        return min(candidates, key=lambda seed: float(seed.length_cost)) if candidates else None
    raise ValueError(f"Unknown extreme seed kind: {kind}")

def _select_diverse_risky_seed(
    seeds: list[SeedEntry],
    *,
    safe_seed: SeedEntry | None,
    top_n: int = 5,
) -> SeedEntry | None:
    """
    Try to find a short trajectory that is actually different 
    from our safest one. Good for diversity.
    """
    candidates = [seed for seed in seeds if seed.length_cost is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda seed: float(seed.length_cost))
    shortlist = candidates[: max(1, min(int(top_n), len(candidates)))]
    if safe_seed is None:
        return shortlist[0]
    # Pick a risky seed that is far away from the safe one in trajectory space
    return max(
        shortlist,
        key=lambda seed: trajectory_distance(np.asarray(seed.trajectory), np.asarray(safe_seed.trajectory)),
    )

def _select_diverse_risky_record(
    records: list[dict[str, object]],
    *,
    safe_record: dict[str, object] | None,
    top_n: int = 5,
) -> dict[str, object] | None:
    """Same as above but works on raw record dicts instead of SeedEntry objects."""
    if not records:
        return None
    sorted_by_length = sorted(records, key=lambda rec: (float(rec["length_cost"]), float(rec["obstacle_cost"])))
    shortlist = sorted_by_length[: max(1, min(int(top_n), len(sorted_by_length)))]
    if safe_record is None:
        return shortlist[0]
    safe_traj = np.asarray(safe_record["trajectory"], dtype=np.float32)
    return max(shortlist, key=lambda rec: trajectory_distance(np.asarray(rec["trajectory"], dtype=np.float32), safe_traj))

def _seed_entry_from_record(rec: dict[str, object], *, default_task_id: str) -> SeedEntry:
    """Convert a record dictionary into a SeedEntry object."""
    traj = np.asarray(rec["trajectory"], dtype=np.float32)
    return SeedEntry(
        trajectory=traj,
        start_config=np.asarray(traj[0], dtype=np.float32),
        goal_config=np.asarray(traj[-1], dtype=np.float32),
        task_id=str(rec.get("task_id", default_task_id)),
        alpha=float(rec.get("alpha")) if rec.get("alpha") is not None else None,
        length_cost=float(rec.get("length_cost")) if rec.get("length_cost") is not None else None,
        obstacle_cost=float(rec.get("obstacle_cost")) if rec.get("obstacle_cost") is not None else None,
    )

def _promote_task_seeds(
    *,
    seed_bank_by_family: dict[str, list[SeedEntry]],
    family: str,
    successful_records: list[dict[str, object]],
    default_task_id: str,
) -> tuple[SeedEntry, SeedEntry] | None:
    """
    Take the best trajectories from a task and add them to the family bank.
    Returns the (safe, risky) pair we just promoted.
    """
    if not successful_records:
        return None
    best_safe = min(successful_records, key=lambda rec: (float(rec["obstacle_cost"]), float(rec["length_cost"])))
    best_risky = _select_diverse_risky_record(successful_records, safe_record=best_safe) or best_safe
    safe_seed_entry = _seed_entry_from_record(best_safe, default_task_id=default_task_id)
    risky_seed_entry = _seed_entry_from_record(best_risky, default_task_id=default_task_id)
    _maybe_add_family_seed(seed_bank_by_family, family=str(family), candidate=safe_seed_entry)
    _maybe_add_family_seed(seed_bank_by_family, family=str(family), candidate=risky_seed_entry)
    return safe_seed_entry, risky_seed_entry

def _adapt_seed_trajectory(
    seed: SeedEntry,
    *,
    start_config: np.ndarray,
    goal_config: np.ndarray,
) -> np.ndarray:
    """
    Math magic to shift an old trajectory so it lines up with a new start/goal.
    We just apply a linear shift across the whole path.
    """
    base = np.asarray(seed.trajectory, dtype=np.float32)
    if base.ndim != 2 or base.shape[0] < 2 or base.shape[1] != start_config.shape[0]:
        raise ValueError("Seed trajectory has an unexpected shape.")
    delta_s = start_config.astype(np.float32) - seed.start_config.astype(np.float32)
    delta_g = goal_config.astype(np.float32) - seed.goal_config.astype(np.float32)
    n = int(base.shape[0])
    fractions = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(n, 1)
    shift = (1.0 - fractions) * delta_s.reshape(1, -1) + fractions * delta_g.reshape(1, -1)
    adapted = (base + shift).astype(np.float32, copy=True)
    adapted[0] = start_config.astype(np.float32)
    adapted[-1] = goal_config.astype(np.float32)
    return adapted

def _maybe_add_family_seed(
    seed_bank_by_family: dict[str, list[SeedEntry]],
    *,
    family: str,
    candidate: SeedEntry,
    distinct_tol: float = SEED_DISTINCT_TOL,
    max_seeds: int = MAX_SEEDS_PER_FAMILY,
) -> None:
    """
    Try to add a new seed to the family bank. 
    If it's too similar to one we already have, we only keep it if it's better.
    If the bank is full, we kick out the most redundant one.
    """
    seeds = seed_bank_by_family.setdefault(str(family), [])
    
    # Check for near-duplicates
    for idx, existing in enumerate(list(seeds)):
        if trajectory_distance(np.asarray(candidate.trajectory), np.asarray(existing.trajectory)) < float(distinct_tol):
            better_safe = (
                candidate.obstacle_cost is not None
                and existing.obstacle_cost is not None
                and float(candidate.obstacle_cost) < float(existing.obstacle_cost) - 1e-6
            )
            better_risky = (
                candidate.length_cost is not None
                and existing.length_cost is not None
                and float(candidate.length_cost) < float(existing.length_cost) - 1e-6
            )
            if better_safe or better_risky:
                seeds[idx] = candidate
            return

    if len(seeds) < int(max_seeds):
        seeds.append(candidate)
        return

    # Bank is full: evict the most redundant non-protected seed
    protected: set[int] = set()
    safe_seed = _select_extreme_seed(seeds, kind="safe")
    risky_seed = _select_extreme_seed(seeds, kind="risky")
    if safe_seed is not None:
        for idx, seed in enumerate(seeds):
            if seed is safe_seed:
                protected.add(idx)
                break
    if risky_seed is not None:
        for idx, seed in enumerate(seeds):
            if seed is risky_seed:
                protected.add(idx)
                break

    min_neighbor_distance: list[tuple[float, int]] = []
    for i, seed in enumerate(seeds):
        if i in protected:
            continue
        best = float("inf")
        for j, other in enumerate(seeds):
            if i == j:
                continue
            d = trajectory_distance(np.asarray(seed.trajectory), np.asarray(other.trajectory))
            if d < best:
                best = float(d)
        min_neighbor_distance.append((best, i))

    if not min_neighbor_distance:
        return
    _, evict_idx = min(min_neighbor_distance, key=lambda item: item[0])
    seeds[evict_idx] = candidate

def save_seed_bank(
    seed_bank_by_family: dict[str, list[SeedEntry]],
    seed_bank_dir: Path,
    n_waypoints: int | None = None,
) -> None:
    """Write the seed bank to disk as .npz files (trajectories) and a manifest.json (metadata)."""
    seed_bank_dir.mkdir(parents=True, exist_ok=True)
    existing = load_seed_bank(seed_bank_dir, n_waypoints=n_waypoints)
    merged: dict[str, list[SeedEntry]] = {family: list(seeds) for family, seeds in existing.items()}
    for family, new_seeds in seed_bank_by_family.items():
        for seed in new_seeds:
            _maybe_add_family_seed(merged, family=family, candidate=seed)
    manifest: dict[str, list[dict[str, object]]] = {}
    for family, seeds in merged.items():
        if not seeds:
            continue
        arrays: dict[str, np.ndarray] = {}
        for i, seed in enumerate(seeds):
            arrays[f"traj_{i}"] = np.asarray(seed.trajectory, dtype=np.float32)
        arrays["start_configs"] = np.stack([np.asarray(s.start_config, dtype=np.float32) for s in seeds])
        arrays["goal_configs"] = np.stack([np.asarray(s.goal_config, dtype=np.float32) for s in seeds])
        np.savez(seed_bank_dir / f"{family}.npz", **arrays)
        manifest[family] = [
            {
                "task_id": seed.task_id,
                "alpha": seed.alpha,
                "length_cost": seed.length_cost,
                "obstacle_cost": seed.obstacle_cost,
            }
            for seed in seeds
        ]
    (seed_bank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def load_seed_bank(
    seed_bank_dir: Path,
    n_waypoints: int | None = None,
) -> dict[str, list[SeedEntry]]:
    """Read the seed bank from disk. Filters by waypoint count if requested."""
    manifest_path = seed_bank_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest: dict[str, list[dict[str, object]]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, list[SeedEntry]] = {}
    for family, seed_metas in manifest.items():
        npz_path = seed_bank_dir / f"{family}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        start_configs = data["start_configs"]
        goal_configs = data["goal_configs"]
        seeds: list[SeedEntry] = []
        for i, meta in enumerate(seed_metas):
            traj_key = f"traj_{i}"
            if traj_key not in data:
                continue
            entry = SeedEntry(
                trajectory=data[traj_key].astype(np.float32),
                start_config=start_configs[i].astype(np.float32),
                goal_config=goal_configs[i].astype(np.float32),
                task_id=str(meta.get("task_id", "")),
                alpha=float(meta["alpha"]) if meta.get("alpha") is not None else None,
                length_cost=float(meta["length_cost"]) if meta.get("length_cost") is not None else None,
                obstacle_cost=float(meta["obstacle_cost"]) if meta.get("obstacle_cost") is not None else None,
            )
            if n_waypoints is not None and entry.trajectory.shape[0] != n_waypoints:
                continue
            seeds.append(entry)
        if seeds:
            result[family] = seeds
    return result
