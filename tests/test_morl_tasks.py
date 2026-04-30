import unittest

import numpy as np

from src.morl.tasks import (
    BASE_TARGET_POSITION,
    MIN_SUCCESSFUL_TASKS_PER_FAMILY,
    NONCONVEX_FAMILIES,
    TaskFamilyConfig,
    TaskSampler,
    WORKSPACE_ANCHOR,
    default_family_mix,
    generate_alpha_values,
    parse_family_mix,
    split_successful_task_ids_by_family,
    split_task_ids,
)


class TaskSamplerTests(unittest.TestCase):
    def test_sampler_is_deterministic(self):
        sampler_a = TaskSampler(seed=123)
        sampler_b = TaskSampler(seed=123)
        tasks_a = sampler_a.sample_tasks(3)
        tasks_b = sampler_b.sample_tasks(3)
        self.assertEqual([task.to_dict() for task in tasks_a], [task.to_dict() for task in tasks_b])

    def test_targets_stay_in_reasonable_bounds(self):
        sampler = TaskSampler(seed=7)
        task = sampler.sample_task(0)
        target = np.asarray(task.target_position)
        self.assertLessEqual(abs(float(target[0] - BASE_TARGET_POSITION[0])), 0.12)
        self.assertLessEqual(abs(float(target[1] - BASE_TARGET_POSITION[1])), 0.12)
        self.assertLessEqual(abs(float(target[2] - BASE_TARGET_POSITION[2])), 0.08)

    def test_sampler_generates_requested_family(self):
        config = TaskFamilyConfig(task_family="corridor")
        sampler = TaskSampler(seed=9, family_config=config)
        task = sampler.sample_task(0)
        self.assertEqual(task.family, "corridor_left_right")
        self.assertGreaterEqual(len(task.obstacles), 3)

    def test_nonconvex_sampler_generates_requested_family(self):
        config = TaskFamilyConfig(task_family="pinch_bottleneck", geometry_regime="nonconvex")
        sampler = TaskSampler(seed=11, family_config=config)
        task = sampler.sample_task(0)
        self.assertEqual(task.family, "pinch_bottleneck")
        self.assertEqual(task.geometry_regime, "nonconvex")
        self.assertGreaterEqual(len(task.obstacles), 4)

    def test_alpha_schedule_generation(self):
        linear = generate_alpha_values(alpha_count=5, schedule="linear")
        dense_middle = generate_alpha_values(alpha_count=5, schedule="dense-middle")
        self.assertEqual(linear[0], 0.0)
        self.assertEqual(linear[-1], 1.0)
        self.assertEqual(len(dense_middle), 5)
        self.assertAlmostEqual(dense_middle[2], 0.5, places=6)

    def test_family_mix_normalizes(self):
        mix = parse_family_mix("corridor=2,pinch=1,stacked=1,asymmetric=0")
        total = sum(weight for _, weight in mix)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_profile_and_regime_defaults(self):
        mix = default_family_mix("max_favoring", "nonconvex")
        family_names = [name for name, _ in mix]
        self.assertEqual(tuple(family_names), NONCONVEX_FAMILIES)
        self.assertAlmostEqual(sum(weight for _, weight in mix), 1.0, places=6)

    def test_nonconvex_tasks_use_nonconvex_families(self):
        config = TaskFamilyConfig(benchmark_profile="max_favoring", geometry_regime="nonconvex")
        sampler = TaskSampler(seed=15, family_config=config)
        tasks = sampler.sample_tasks(5)
        self.assertTrue(all(task.family in NONCONVEX_FAMILIES for task in tasks))
        self.assertTrue(all(task.benchmark_profile == "max_favoring" for task in tasks))

    def test_sampler_pushes_obstacles_off_centerline(self):
        config = TaskFamilyConfig(task_family="asymmetric", difficulty="medium")
        sampler = TaskSampler(seed=21, family_config=config)
        task = sampler.sample_task(0)
        target = np.asarray(task.target_position, dtype=np.float64)
        distances = [
            sampler._line_distance(np.asarray(obstacle.center, dtype=np.float64), WORKSPACE_ANCHOR, target)
            for obstacle in task.obstacles
        ]
        self.assertGreater(min(distances), 0.02)
        max_clearance = max(obstacle.radius + obstacle.safe_distance for obstacle in task.obstacles)
        self.assertLess(max_clearance, 0.09)

    def test_split_manifest_covers_every_task(self):
        task_ids = [f"task_{index}" for index in range(10)]
        splits = split_task_ids(task_ids, seed=5)
        merged = splits["train"] + splits["val"] + splits["test"]
        self.assertEqual(sorted(merged), sorted(task_ids))

    def test_family_stratified_split_enforces_floor(self):
        payload = split_successful_task_ids_by_family(
            {
                "double_corridor": [f"dc_{index}" for index in range(7)],
                "offset_gate": [f"og_{index}" for index in range(9)],
            },
            seed=3,
        )
        self.assertEqual(len(payload["split_by_family"]["train"]["double_corridor"]), 5)
        self.assertEqual(len(payload["split_by_family"]["val"]["double_corridor"]), 1)
        self.assertEqual(len(payload["split_by_family"]["test"]["double_corridor"]), 1)
        merged = payload["splits"]["train"] + payload["splits"]["val"] + payload["splits"]["test"]
        self.assertEqual(len(merged), 16)
        self.assertEqual(len(set(merged)), 16)

    def test_family_stratified_split_rejects_insufficient_support(self):
        with self.assertRaisesRegex(ValueError, f"at least {MIN_SUCCESSFUL_TASKS_PER_FAMILY}"):
            split_successful_task_ids_by_family(
                {
                    "double_corridor": [f"dc_{index}" for index in range(6)],
                },
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
