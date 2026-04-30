import json
import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.morl.collect_dataset import parse_args
from src.morl.semantics import artifact_semantics_payload


def _write_dataset(root: Path, seed_name: str, mode: str, records: list[dict]) -> None:
    dataset_dir = root / mode / seed_name
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    benchmark_profile = records[0]["task_spec"].get("benchmark_profile", "baseline")
    geometry_regime = records[0]["task_spec"].get("geometry_regime", "mixed")
    task_family = records[0]["task_spec"].get("family", "corridor_left_right")
    metadata = []
    for record in records:
        with (raw_dir / f"{record['trajectory_id']}.pkl").open("wb") as handle:
            pickle.dump(record, handle)
        metadata.append(
            {
                "trajectory_id": record["trajectory_id"],
                "task_id": record["task_spec"]["task_id"],
                "alpha": record["alpha"],
                "length_cost": record["length_cost"],
                "obstacle_cost": record["obstacle_cost"],
                "filename": f"{record['trajectory_id']}.pkl",
            }
        )
    (dataset_dir / "trajectory_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = {
        "planner_mode": mode,
        "seed": int(seed_name.split("_")[-1]),
        "task_family": task_family,
        "benchmark_profile": benchmark_profile,
        "geometry_regime": geometry_regime,
        "nonconvex_route_count": 2 if geometry_regime == "nonconvex" else 0,
        **artifact_semantics_payload(),
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


class BenchmarkReportTests(unittest.TestCase):
    def test_collect_dataset_profile_defaults(self):
        argv = [
            "collect_dataset.py",
            "--experiment-name",
            "tmp",
            "--planner-mode",
            "max",
            "--benchmark-profile",
            "max_favoring",
            "--geometry-regime",
            "nonconvex",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.difficulty, "hard")
        self.assertEqual(args.alpha_count, 15)
        self.assertEqual(args.alpha_schedule, "dense-middle")
        self.assertEqual(args.restart_count, 5)

    def test_compare_benchmarks_generates_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / "test_artifacts" / "benchmark_tmp"
        if root.exists():
            import shutil

            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            base_task = {
                "task_id": "task_0000",
                "planner_seed": 1,
                "start_config": [0.0] * 7,
                "target_position": [0.65, 0.0, 0.2],
                "obstacles": [],
                "family": "pinch_bottleneck",
                "difficulty": "medium",
                "benchmark_profile": "max_favoring",
                "geometry_regime": "nonconvex",
                "horizon": 25,
                "dt": 0.1,
            }
            sum_records = [
                {
                    "trajectory_id": "sum_a",
                    "trajectory": np.zeros((3, 7), dtype=np.float32),
                    "alpha": 0.0,
                    "length_cost": 2.0,
                    "obstacle_cost": 2.0,
                    "task_spec": base_task,
                }
            ]
            max_records = [
                {
                    "trajectory_id": "max_a",
                    "trajectory": np.zeros((3, 7), dtype=np.float32),
                    "alpha": 0.0,
                    "length_cost": 1.0,
                    "obstacle_cost": 2.0,
                    "task_spec": base_task,
                },
                {
                    "trajectory_id": "max_b",
                    "trajectory": np.ones((3, 7), dtype=np.float32),
                    "alpha": 0.5,
                    "length_cost": 2.0,
                    "obstacle_cost": 1.0,
                    "task_spec": base_task,
                },
            ]
            _write_dataset(root, "seed_0001", "sum", sum_records)
            _write_dataset(root, "seed_0001", "max", max_records)

            output_dir = root / "report"
            command = [
                sys.executable,
                "-m",
                "src.morl.compare_benchmarks",
                "--sum-dir",
                str(root / "sum"),
                "--max-dir",
                str(root / "max"),
                "--output-dir",
                str(output_dir),
                "--size-matched-samples",
                "8",
            ]
            subprocess.check_call(command, cwd=repo_root)

            report = json.loads((output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["seed_count"], 1)
            self.assertGreater(report["paired_rows"][0]["unique_count_diff"], 0)
            self.assertIn("pinch_bottleneck", report["family_summary"])
            self.assertIn("nonconvex_summary", report)
            self.assertEqual(report["paired_rows"][0]["geometry_regime"], "nonconvex")
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_compare_benchmarks_rejects_profile_mismatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / "test_artifacts" / "benchmark_mismatch_tmp"
        if root.exists():
            import shutil

            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            sum_task = {
                "task_id": "task_0000",
                "planner_seed": 1,
                "start_config": [0.0] * 7,
                "target_position": [0.65, 0.0, 0.2],
                "obstacles": [],
                "family": "asymmetric_safe_margin",
                "difficulty": "medium",
                "benchmark_profile": "baseline",
                "geometry_regime": "convex",
                "horizon": 25,
                "dt": 0.1,
            }
            max_task = dict(sum_task)
            max_task["benchmark_profile"] = "max_favoring"
            sum_records = [{"trajectory_id": "sum_a", "trajectory": np.zeros((3, 7), dtype=np.float32), "alpha": 0.0, "length_cost": 2.0, "obstacle_cost": 2.0, "task_spec": sum_task}]
            max_records = [{"trajectory_id": "max_a", "trajectory": np.zeros((3, 7), dtype=np.float32), "alpha": 0.0, "length_cost": 1.0, "obstacle_cost": 1.0, "task_spec": max_task}]
            _write_dataset(root, "seed_0001", "sum", sum_records)
            _write_dataset(root, "seed_0001", "max", max_records)

            command = [
                sys.executable,
                "-m",
                "src.morl.compare_benchmarks",
                "--sum-dir",
                str(root / "sum"),
                "--max-dir",
                str(root / "max"),
            ]
            result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Benchmark profile or geometry regime mismatch", result.stderr)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
