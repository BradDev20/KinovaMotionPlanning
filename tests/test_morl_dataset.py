import unittest
from pathlib import Path

import numpy as np

from src.morl.dataset import (
    build_transition_dataset,
    deduplicate_records,
    record_to_transition_arrays,
    summarize_records,
    trajectory_distance,
)
from src.morl.tasks import TaskSampler

try:
    import mujoco  # noqa: F401
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


class DatasetTests(unittest.TestCase):
    def test_deduplicate_records(self):
        task = TaskSampler(seed=4).sample_task(0)
        base_record = {
            "trajectory_id": "a",
            "trajectory": np.zeros((3, 7), dtype=np.float32),
            "length_cost": 1.0,
            "obstacle_cost": 2.0,
            "task_spec": task.to_dict(),
        }
        duplicate = {
            "trajectory_id": "b",
            "trajectory": np.zeros((3, 7), dtype=np.float32),
            "length_cost": 1.0001,
            "obstacle_cost": 2.0001,
            "task_spec": task.to_dict(),
        }
        unique = deduplicate_records([base_record, duplicate], objective_tol=1e-2, path_tol=1e-3)
        self.assertEqual(len(unique), 1)

    def test_summarize_records_reports_route_and_objective_counts(self):
        task = TaskSampler(seed=5).sample_task(0)
        records = [
            {
                "trajectory_id": "a",
                "trajectory": np.zeros((4, 7), dtype=np.float32),
                "length_cost": 1.0,
                "obstacle_cost": 2.0,
                "task_spec": task.to_dict(),
            },
            {
                "trajectory_id": "b",
                "trajectory": np.ones((4, 7), dtype=np.float32) * 0.5,
                "length_cost": 1.5,
                "obstacle_cost": 1.0,
                "task_spec": task.to_dict(),
            },
        ]
        summary = summarize_records(records, objective_tol=1e-3, path_tol=1e-2)
        self.assertEqual(summary["unique_trajectory_count"], 2)
        self.assertEqual(summary["unique_objective_count"], 2)
        self.assertGreaterEqual(summary["unique_route_count"], 1)
        self.assertIn(task.family, summary["family_breakdown"])

    def test_build_transition_dataset_allows_empty_records(self):
        artifact_dir = Path("test_artifacts") / "dataset_tmp"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_dir / "transitions_empty.npz"
        if output_path.exists():
            output_path.unlink()
        summary = build_transition_dataset([], scene_dir=artifact_dir / "scenes", output_path=output_path)

        self.assertEqual(summary["transition_count"], 0)
        self.assertEqual(summary["trajectory_count"], 0)
        self.assertEqual(summary["observation_dim"], 0)
        self.assertEqual(summary["action_dim"], 0)
        self.assertTrue(output_path.exists())

    def test_trajectory_distance_handles_empty_samples(self):
        lhs = np.zeros((0, 7), dtype=np.float32)
        rhs = np.zeros((3, 7), dtype=np.float32)
        self.assertEqual(trajectory_distance(lhs, rhs), 0.0)

    @unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
    def test_record_to_transition_arrays_shapes(self):
        task = TaskSampler(seed=11).sample_task(0)
        trajectory = np.vstack(
            [
                np.asarray(task.start_config, dtype=np.float32),
                np.asarray(task.start_config, dtype=np.float32) + 0.01,
                np.asarray(task.start_config, dtype=np.float32) + 0.02,
            ]
        )
        record = {
            "trajectory_id": "demo",
            "trajectory": trajectory,
            "alpha": 0.5,
            "length_weight": 0.5,
            "obstacle_weight": 0.5,
            "planner_mode": "sum",
            "task_spec": task.to_dict(),
        }
        artifact_root = Path("test_artifacts") / "dataset_shapes_tmp"
        artifact_root.mkdir(parents=True, exist_ok=True)
        arrays = record_to_transition_arrays(record, scene_dir=artifact_root / "scenes")
        self.assertEqual(arrays["observations"].shape[0], 2)
        self.assertEqual(arrays["actions"].shape, (2, 7))
        self.assertEqual(arrays["reward_vectors"].shape, (2, 2))
        self.assertEqual(int(arrays["dones"].sum()), 1)
        self.assertFalse(bool(arrays["dones"][0]))
        self.assertTrue(bool(arrays["dones"][-1]))


if __name__ == "__main__":
    unittest.main()
