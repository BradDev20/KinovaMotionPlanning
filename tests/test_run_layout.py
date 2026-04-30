from __future__ import annotations

import unittest
from pathlib import Path

from src.morl.run_layout import (
    checkpoint_dir_for_run,
    compare_dir_for_run,
    dataset_dir_for_run,
    default_compare_output_dir,
    default_evaluation_output_dir,
    default_training_output_dir,
    evaluation_dir_for_run,
    infer_dataset_dir_from_checkpoint,
    is_flattened_dataset_dir,
    pipeline_summary_path_for_run,
)


class RunLayoutTests(unittest.TestCase):
    def test_flattened_run_helpers(self):
        root = Path("data/runs")
        self.assertEqual(dataset_dir_for_run("demo_run", "max", root=root), root / "demo_run" / "max" / "dataset")
        self.assertEqual(checkpoint_dir_for_run("demo_run", "max", root=root), root / "demo_run" / "max" / "checkpoints" / "max_iql")
        self.assertEqual(evaluation_dir_for_run("demo_run", "max", root=root), root / "demo_run" / "max" / "evaluation")
        self.assertEqual(pipeline_summary_path_for_run("demo_run", "max", root=root), root / "demo_run" / "max" / "pipeline_summary.json")
        self.assertEqual(compare_dir_for_run("demo_run", root=root), root / "demo_run" / "compare")

    def test_is_flattened_dataset_dir(self):
        self.assertTrue(is_flattened_dataset_dir(Path("data/runs/demo/max/dataset")))
        self.assertFalse(is_flattened_dataset_dir(Path("data/morl/demo/max")))

    def test_default_training_output_dir_uses_sibling_checkpoints_for_flattened_layout(self):
        dataset_dir = Path("data/runs/demo/max/dataset")
        self.assertEqual(default_training_output_dir(dataset_dir, "max"), Path("data/runs/demo/max/checkpoints/max_iql"))

    def test_default_training_output_dir_keeps_legacy_layout(self):
        dataset_dir = Path("data/morl/demo/max")
        self.assertEqual(default_training_output_dir(dataset_dir, "max"), Path("data/morl/demo/max/checkpoints/max_iql"))

    def test_infer_dataset_dir_from_checkpoint_supports_flattened_layout(self):
        artifact_root = Path("test_artifacts") / "flattened_layout_tmp" / "demo_run" / "max"
        dataset_dir = artifact_root / "dataset"
        checkpoint_path = artifact_root / "checkpoints" / "max_iql" / "checkpoint.pt"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"")

        self.assertEqual(infer_dataset_dir_from_checkpoint(checkpoint_path), dataset_dir)

    def test_infer_dataset_dir_from_checkpoint_supports_legacy_layout(self):
        checkpoint_path = Path("data/morl/demo/max/checkpoints/max_iql/checkpoint.pt")
        self.assertEqual(infer_dataset_dir_from_checkpoint(checkpoint_path), Path("data/morl/demo/max"))

    def test_default_evaluation_output_dir_uses_mode_root_for_flattened_layout(self):
        checkpoint_path = Path("data/runs/demo/max/checkpoints/max_iql/checkpoint.pt")
        dataset_dir = Path("data/runs/demo/max/dataset")
        self.assertEqual(default_evaluation_output_dir(checkpoint_path, dataset_dir), Path("data/runs/demo/max/evaluation"))

    def test_default_evaluation_output_dir_keeps_legacy_layout(self):
        checkpoint_path = Path("data/morl/demo/max/checkpoints/max_iql/checkpoint.pt")
        dataset_dir = Path("data/morl/demo/max")
        self.assertEqual(default_evaluation_output_dir(checkpoint_path, dataset_dir), Path("data/morl/demo/max/checkpoints/max_iql/evaluation"))

    def test_default_compare_output_dir_uses_run_compare_dir_for_flattened_layout(self):
        sum_dir = Path("data/runs/demo/sum/dataset")
        max_dir = Path("data/runs/demo/max/dataset")
        self.assertEqual(default_compare_output_dir(sum_dir, max_dir), Path("data/runs/demo/compare"))

    def test_default_compare_output_dir_keeps_legacy_layout(self):
        sum_dir = Path("data/morl/demo/sum")
        max_dir = Path("data/morl/demo/max")
        self.assertEqual(default_compare_output_dir(sum_dir, max_dir), Path("data/morl/demo/benchmark_report"))


if __name__ == "__main__":
    unittest.main()
