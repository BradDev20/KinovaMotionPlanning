from __future__ import annotations

from typing import Iterable

import numpy as np

from src.numba_compat import numba_njit

try:
    import torch
except ImportError:  # pragma: no cover - handled in runtime checks
    torch = None


def alpha_to_weights(alpha: float) -> np.ndarray:
    clipped = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray([clipped, 1.0 - clipped], dtype=np.float32)


def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(weights), dtype=np.float32)
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return array / total


@numba_njit(cache=True)
def _scalarize_sum_kernel(array: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    rows = array.shape[0]
    cols = array.shape[1]
    out = np.empty(rows, dtype=np.float32)
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += float(array[i, j]) * float(normalized[j])
        out[i] = total
    return out


@numba_njit(cache=True)
def _scalarize_max_kernel(array: np.ndarray, normalized: np.ndarray, rho: float) -> np.ndarray:
    rows = array.shape[0]
    cols = array.shape[1]
    out = np.empty(rows, dtype=np.float32)
    for i in range(rows):
        max_value = -1e30
        raw_sum = 0.0
        for j in range(cols):
            weighted = float(array[i, j]) * float(normalized[j])
            if weighted > max_value:
                max_value = weighted
            raw_sum += float(array[i, j])
        out[i] = max_value + float(rho) * raw_sum
    return out


@numba_njit(cache=True)
def _pareto_front_sorted(sorted_points: np.ndarray) -> tuple[np.ndarray, int]:
    front = np.empty_like(sorted_points)
    count = 0
    best_y = np.inf
    for i in range(sorted_points.shape[0]):
        y_value = float(sorted_points[i, 1])
        if y_value <= best_y:
            front[count, 0] = sorted_points[i, 0]
            front[count, 1] = sorted_points[i, 1]
            best_y = y_value
            count += 1
    return front, count


@numba_njit(cache=True)
def _hypervolume_from_front(front: np.ndarray, ref_x: float, ref_y: float) -> float:
    volume = 0.0
    previous_y = ref_y
    for i in range(front.shape[0]):
        x_value = float(front[i, 0])
        y_value = float(front[i, 1])
        width = ref_x - x_value
        if width < 0.0:
            width = 0.0
        height = previous_y - y_value
        if height < 0.0:
            height = 0.0
        volume += width * height
        if y_value < previous_y:
            previous_y = y_value
    return volume


def scalarize_numpy(
    values: np.ndarray,
    weights: Iterable[float],
    mode: str,
    rho: float = 0.01,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    normalized = normalize_weights(weights)
    if array.shape[-1] != normalized.shape[0]:
        raise ValueError("Value vector and weights must have the same final dimension.")

    if mode == "sum":
        return _scalarize_sum_kernel(array, normalized)
    if mode == "max":
        return _scalarize_max_kernel(array, normalized, float(rho))
    raise ValueError(f"Unsupported scalarization mode: {mode}")


def scalarize_torch(
    values,
    weights,
    mode: str,
    rho: float = 0.01,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for torch scalarization.")

    if values.shape[-1] != weights.shape[-1]:
        raise ValueError("Value tensor and weight tensor must have the same final dimension.")

    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = weights / weights_sum
    if mode == "sum":
        return (values * normalized).sum(dim=-1)
    if mode == "max":
        return (values * normalized).max(dim=-1).values + float(rho) * values.sum(dim=-1)
    raise ValueError(f"Unsupported scalarization mode: {mode}")


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset for 2D minimization."""
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 2)
    order = np.argsort(array[:, 0])
    sorted_points = array[order]
    front, count = _pareto_front_sorted(sorted_points)
    return np.asarray(front[:count], dtype=np.float64)


def hypervolume_2d(points: np.ndarray, reference: tuple[float, float]) -> float:
    """Simple dominated hypervolume for 2D minimization."""
    front = pareto_front(points)
    if front.size == 0:
        return 0.0

    ref_x, ref_y = float(reference[0]), float(reference[1])
    return float(_hypervolume_from_front(front, ref_x, ref_y))
