import unittest

import numpy as np

from src.morl.scalarization import alpha_to_weights, hypervolume_2d, pareto_front, scalarize_numpy, scalarize_torch

try:
    import torch
except ImportError:
    torch = None

try:
    from src.motion_planning.cost_functions import ObstacleAvoidanceCostFunction
    from src.motion_planning.utils import Obstacle
except ModuleNotFoundError:
    ObstacleAvoidanceCostFunction = None
    Obstacle = None


class _StubKinematics:
    def __init__(self, positions: list[np.ndarray]):
        self.positions = [np.asarray(position, dtype=np.float64) for position in positions]
        self.index = 0

    def _backup_state(self):
        self._backup_index = self.index

    def _restore_state(self):
        self.index = getattr(self, "_backup_index", self.index)

    def forward_kinematics(self, joint_positions):
        position = self.positions[self.index]
        self.index += 1
        return position, np.eye(3)


class ScalarizationTests(unittest.TestCase):
    def test_alpha_to_weights(self):
        weights = alpha_to_weights(0.25)
        self.assertTrue(np.allclose(weights, np.array([0.25, 0.75], dtype=np.float32)))

    def test_weighted_sum(self):
        values = np.array([[1.0, 3.0]], dtype=np.float32)
        result = scalarize_numpy(values, [0.25, 0.75], "sum")
        self.assertAlmostEqual(float(result[0]), 2.5)

    def test_weighted_max(self):
        values = np.array([[2.0, 5.0]], dtype=np.float32)
        result = scalarize_numpy(values, [0.6, 0.4], "max", rho=0.1)
        expected = max(0.6 * 2.0, 0.4 * 5.0) + 0.1 * (2.0 + 5.0)
        self.assertAlmostEqual(float(result[0]), expected, places=6)

    def test_torch_scalarization_matches_numpy(self):
        if torch is None:
            self.skipTest("PyTorch is not installed.")
        values = np.array([[1.5, 2.5], [2.0, 1.0]], dtype=np.float32)
        weights = np.array([[0.25, 0.75], [0.6, 0.4]], dtype=np.float32)
        numpy_sum = scalarize_numpy(values, weights[0], "sum")
        torch_sum = scalarize_torch(
            torch.as_tensor(values[:1], dtype=torch.float32),
            torch.as_tensor(weights[:1], dtype=torch.float32),
            "sum",
        )
        self.assertAlmostEqual(float(torch_sum[0].item()), float(numpy_sum[0]), places=6)

        numpy_max = scalarize_numpy(values[1:], weights[1], "max", rho=0.1)
        torch_max = scalarize_torch(
            torch.as_tensor(values[1:], dtype=torch.float32),
            torch.as_tensor(weights[1:], dtype=torch.float32),
            "max",
            rho=0.1,
        )
        self.assertAlmostEqual(float(torch_max[0].item()), float(numpy_max[0]), places=6)

    def test_pareto_front_and_hypervolume(self):
        points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 4.0], [1.5, 2.5]])
        front = pareto_front(points)
        self.assertEqual(front.shape[0], 3)
        self.assertGreater(hypervolume_2d(points, reference=(4.0, 5.0)), 0.0)

    def test_obstacle_aggregate_max_differs_from_avg(self):
        if ObstacleAvoidanceCostFunction is None or Obstacle is None:
            self.skipTest("mujoco-dependent motion_planning imports are unavailable in this interpreter.")
        trajectory = np.zeros((3, 7), dtype=np.float64)
        positions = [
            np.array([0.35, 0.0, 0.2], dtype=np.float64),
            np.array([0.49, 0.0, 0.2], dtype=np.float64),
            np.array([0.35, 0.0, 0.2], dtype=np.float64),
        ]
        obstacle = Obstacle(center=np.array([0.50, 0.0, 0.2]), radius=0.03, safe_distance=0.04)
        avg_cost = ObstacleAvoidanceCostFunction(
            kinematics_solver=_StubKinematics(positions),
            obstacles=[obstacle],
            aggregate="avg",
            decay_rate=8.0,
        )
        max_cost = ObstacleAvoidanceCostFunction(
            kinematics_solver=_StubKinematics(positions),
            obstacles=[obstacle],
            aggregate="max",
            decay_rate=8.0,
        )
        avg_value = avg_cost.compute_cost(trajectory, dt=0.1)
        max_value = max_cost.compute_cost(trajectory, dt=0.1)
        self.assertGreater(max_value, avg_value)


if __name__ == "__main__":
    unittest.main()
