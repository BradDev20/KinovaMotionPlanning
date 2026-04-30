import unittest
from types import MethodType

import numpy as np

from src.motion_planning.cost_functions import (
    MuJoCoRobotObstacleCost,
)


def _build_cost(*, aggregate: str, weight: float, uncached_fn):
    cost = MuJoCoRobotObstacleCost.__new__(MuJoCoRobotObstacleCost)
    cost.weight = float(weight)
    cost.aggregate = aggregate
    cost._waypoint_cost_cache = {}
    cost._waypoint_cost_cache_decimals = 5
    cost._waypoint_cost_cache_size = 1000
    cost._compute_configuration_cost_uncached = MethodType(uncached_fn, cost)
    return cost


def _expected_gradient_from_full_recompute(cost, trajectory: np.ndarray, dt: float) -> np.ndarray:
    eps = 1e-5
    base = cost.compute_cost(trajectory.copy(), dt)
    expected = np.zeros_like(trajectory)
    for i in range(trajectory.shape[0]):
        for j in range(trajectory.shape[1]):
            perturbed = trajectory.copy()
            perturbed[i, j] += eps
            expected[i, j] = (cost.compute_cost(perturbed, dt) - base) / eps
    return expected * cost.weight


class MuJoCoRobotObstacleCostCacheTests(unittest.TestCase):
    def test_compute_cost_reuses_cached_waypoint_costs(self):
        calls = []

        def uncached_fn(self, q):
            calls.append(tuple(np.asarray(q, dtype=float)))
            q = np.asarray(q, dtype=float)
            return float(np.sum(q ** 2))

        cost = _build_cost(aggregate="sum", weight=1.0, uncached_fn=uncached_fn)
        trajectory = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=float)

        value = cost.compute_cost(trajectory, dt=0.5)

        self.assertAlmostEqual(value, 17.5)
        self.assertEqual(len(calls), 2)

    def test_compute_gradient_avoids_full_compute_cost_reentry(self):
        def uncached_fn(self, q):
            q = np.asarray(q, dtype=float)
            return float((q[0] ** 2) + (3.0 * q[1] ** 2))

        expected_cost = _build_cost(aggregate="sum", weight=1.5, uncached_fn=uncached_fn)
        actual_cost = _build_cost(aggregate="sum", weight=1.5, uncached_fn=uncached_fn)
        trajectory = np.asarray([[1.0, 2.0], [0.5, -1.0]], dtype=float)
        expected = _expected_gradient_from_full_recompute(expected_cost, trajectory, dt=0.2)

        def fail_compute_cost(self, trajectory, dt=0.1):
            raise AssertionError("compute_cost should not be called from compute_gradient")

        actual_cost.compute_cost = MethodType(fail_compute_cost, actual_cost)
        gradient = actual_cost.compute_gradient(trajectory.copy(), dt=0.2)

        np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-6)

    def test_compute_gradient_matches_full_recompute_for_max_aggregate(self):
        def uncached_fn(self, q):
            q = np.asarray(q, dtype=float)
            return float((2.0 * q[0] ** 2) + (q[1] ** 2))

        expected_cost = _build_cost(aggregate="max", weight=2.0, uncached_fn=uncached_fn)
        actual_cost = _build_cost(aggregate="max", weight=2.0, uncached_fn=uncached_fn)
        trajectory = np.asarray([[0.2, 0.1], [1.5, -0.5], [0.3, 0.2]], dtype=float)

        expected = _expected_gradient_from_full_recompute(expected_cost, trajectory, dt=0.1)
        gradient = actual_cost.compute_gradient(trajectory.copy(), dt=0.1)

        np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-6)



if __name__ == "__main__":
    unittest.main()
