from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from src.cli.__main__ import build_parser, run_pipeline
from src.morl.collect_dataset import CollectionSupportError
from src.morl.run_layout import dataset_dir_for_run, pipeline_summary_path_for_run


class PipelineCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("test_artifacts") / "pipeline_cli_tmp"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.parser = build_parser()

    def _args(self) -> object:
        return self.parser.parse_args(["pipeline", "--run", "demo_run", "--mode", "max", "--root", str(self.root)])

    def test_pipeline_writes_summary_after_success(self):
        args = self._args()
        dataset_dir = dataset_dir_for_run(args.run, args.mode, root=args.root)
        summary_path = pipeline_summary_path_for_run(args.run, args.mode, root=args.root)
        support_check = {
            "passed": True,
            "failure_reason": None,
            "split_task_ids_by_family": {
                "train": {"double_corridor": ["task_0001", "task_0002", "task_0003", "task_0004", "task_0005"]},
                "val": {"double_corridor": ["task_0006"]},
                "test": {"double_corridor": ["task_0007"]},
            },
        }

        def fake_collect(_argv: list[str]) -> dict[str, object]:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dataset_summary.json").write_text(json.dumps({"support_check": support_check}), encoding="utf-8")
            (dataset_dir / "splits.json").write_text(json.dumps({"train": [], "val": [], "test": []}), encoding="utf-8")
            return {"path": str(dataset_dir)}

        with (
            mock.patch("src.morl.collect_dataset.main", side_effect=fake_collect),
            mock.patch("src.morl.train_offline.main") as train_main,
            mock.patch("src.morl.evaluate.main") as eval_main,
        ):
            run_pipeline(args)

        self.assertTrue(summary_path.exists())
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_dir"], str(dataset_dir))
        self.assertEqual(
            payload["successful_train_task_ids_by_family"]["double_corridor"],
            ["task_0001", "task_0002", "task_0003", "task_0004", "task_0005"],
        )
        train_main.assert_called_once()
        eval_main.assert_called_once()

    def test_pipeline_writes_summary_and_stops_after_support_failure(self):
        args = self._args()
        dataset_dir = dataset_dir_for_run(args.run, args.mode, root=args.root)
        summary_path = pipeline_summary_path_for_run(args.run, args.mode, root=args.root)
        support_check = {
            "passed": False,
            "failure_reason": "Family offset_gate has only 3 successful tasks.",
            "split_task_ids_by_family": {"train": {}, "val": {}, "test": {}},
        }

        def fake_collect(_argv: list[str]) -> None:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dataset_summary.json").write_text(json.dumps({"support_check": support_check}), encoding="utf-8")
            raise CollectionSupportError(
                "Family offset_gate has only 3 successful tasks.",
                dataset_dir=dataset_dir,
                support_check=support_check,
            )

        with (
            mock.patch("src.morl.collect_dataset.main", side_effect=fake_collect),
            mock.patch("src.morl.train_offline.main") as train_main,
            mock.patch("src.morl.evaluate.main") as eval_main,
        ):
            with self.assertRaises(CollectionSupportError):
                run_pipeline(args)

        self.assertTrue(summary_path.exists())
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_dir"], str(dataset_dir))
        self.assertEqual(payload["support_check"]["failure_reason"], "Family offset_gate has only 3 successful tasks.")
        train_main.assert_not_called()
        eval_main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
