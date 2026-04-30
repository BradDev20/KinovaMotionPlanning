import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.morl.iql import IQLConfig, PreferenceConditionedIQL, torch
from src.morl.train_offline import infer_action_limit


@unittest.skipIf(torch is None, "PyTorch is not installed")
class IQLTests(unittest.TestCase):
    def test_infer_action_limit_respects_physical_velocity_cap(self):
        dataset = {"actions": np.full((8, 7), 0.5, dtype=np.float32)}

        class _Task:
            dt = 0.1

        with mock.patch("src.morl.train_offline.load_tasks", return_value=[_Task()]):
            action_limit = infer_action_limit(Path("."), dataset, max_joint_velocity=1.3)
        self.assertAlmostEqual(action_limit, 0.13, places=6)

    def test_train_step_returns_metrics(self):
        agent = PreferenceConditionedIQL(
            obs_dim=6,
            action_dim=3,
            config=IQLConfig(hidden_dim=32, action_limit=0.5),
            device="cpu",
        )
        batch = {
            "observations": np.random.randn(16, 6).astype(np.float32),
            "actions": np.random.randn(16, 3).astype(np.float32),
            "next_observations": np.random.randn(16, 6).astype(np.float32),
            "reward_vectors": np.random.randn(16, 2).astype(np.float32),
            "dones": np.zeros(16, dtype=np.float32),
            "weights": np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (16, 1)),
        }
        metrics = agent.train_step(batch)
        self.assertIn("q_loss", metrics)
        self.assertIn("policy_loss", metrics)
        action = agent.act(np.zeros(6, dtype=np.float32), np.array([0.5, 0.5], dtype=np.float32))
        self.assertEqual(action.shape, (3,))


if __name__ == "__main__":
    unittest.main()
