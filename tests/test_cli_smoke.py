from __future__ import annotations

import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from src.cli.__main__ import _build_collect_backend_argv, build_parser


class CliSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_collect_parser(self):
        args = self.parser.parse_args(["collect", "--run", "demo_run", "--mode", "max"])
        self.assertEqual(args.command, "collect")
        self.assertEqual(args.run, "demo_run")
        self.assertEqual(args.mode, "max")
        self.assertEqual(args.planner_steps, 250)
        self.assertEqual(args.gpu_batch_size, 32)

    def test_collect_parser_accepts_torch_options(self):
        args = self.parser.parse_args(
            [
                "collect",
                "--run",
                "demo_run",
                "--mode",
                "max",
                "--device",
                "cpu",
                "--num-workers",
                "2",
                "--planner-steps",
                "12",
                "--repair-max-iter",
                "9",
                "--repair-max-fun",
                "42",
                "--quiet",
            ]
        )
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.num_workers, 2)
        self.assertEqual(args.planner_steps, 12)
        self.assertEqual(args.repair_max_iter, 9)
        self.assertEqual(args.repair_max_fun, 42)
        self.assertTrue(args.quiet)

    def test_collect_parser_accepts_profile_options(self):
        args = self.parser.parse_args(
            [
                "collect",
                "--run",
                "demo_run",
                "--mode",
                "max",
                "--benchmark-profile",
                "max_favoring",
                "--profile",
            ]
        )
        self.assertEqual(args.benchmark_profile, "max_favoring")
        self.assertTrue(args.profile)

    def test_train_parser(self):
        args = self.parser.parse_args(["train", "--run", "demo_run", "--mode", "sum"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.mode, "sum")

    def test_pipeline_parser(self):
        args = self.parser.parse_args(["pipeline", "--run", "demo_run", "--mode", "max", "--quiet"])
        self.assertEqual(args.command, "pipeline")
        self.assertEqual(args.mode, "max")
        self.assertTrue(args.quiet)

    def test_pipeline_collect_backend_argv_forwards_quiet(self):
        args = self.parser.parse_args(["pipeline", "--run", "demo_run", "--mode", "sum", "--quiet"])
        backend_argv = _build_collect_backend_argv(args)
        self.assertIn("--quiet", backend_argv)

    def test_pipeline_collect_backend_argv_forwards_profile(self):
        args = self.parser.parse_args(["pipeline", "--run", "demo_run", "--mode", "sum", "--profile"])
        backend_argv = _build_collect_backend_argv(args)
        self.assertIn("--profile", backend_argv)

    def test_collect_backend_argv_uses_flattened_run_layout_path(self):
        args = self.parser.parse_args(["collect", "--run", "demo_run", "--mode", "max"])
        backend_argv = _build_collect_backend_argv(args)
        dataset_dir = Path(backend_argv[backend_argv.index("--dataset-dir") + 1])
        self.assertEqual(dataset_dir, Path("data/runs/demo_run/max/dataset"))

    def test_gen3_gripper_mesh_paths_resolve_inside_repo(self):
        repo_root = Path(__file__).resolve().parents[1]
        gen3_xml = repo_root / "robot_models" / "kinova_gen3" / "gen3.xml"
        root = ET.fromstring(gen3_xml.read_text(encoding="utf-8"))
        compiler = root.find("./compiler")
        meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
        mesh_base = (gen3_xml.parent / meshdir).resolve() if meshdir else gen3_xml.parent.resolve()
        mesh_files = [
            mesh.attrib["file"]
            for mesh in root.findall("./asset/mesh")
            if mesh.attrib.get("name", "").startswith("gripper_")
        ]
        self.assertTrue(mesh_files)
        for mesh_file in mesh_files:
            resolved = (mesh_base / mesh_file).resolve()
            self.assertTrue(resolved.exists(), msg=f"Missing mesh asset for {mesh_file}: {resolved}")

    def test_eval_parser(self):
        args = self.parser.parse_args(["eval", "--run", "demo_run", "--mode", "max"])
        self.assertEqual(args.command, "eval")
        self.assertEqual(args.mode, "max")

    def test_compare_parser(self):
        args = self.parser.parse_args(["compare", "--run", "demo_run"])
        self.assertEqual(args.command, "compare")
        self.assertEqual(args.run, "demo_run")

    def test_replay_parser(self):
        args = self.parser.parse_args(["replay", "--run", "demo_run", "--mode", "max", "--task", "task_0003"])
        self.assertEqual(args.command, "replay")
        self.assertEqual(args.task, "task_0003")

    def test_replay_multi_task_parser(self):
        args = self.parser.parse_args(["replay", "--run", "demo_run", "--mode", "max", "--tasks", "task_0003", "task_0004"])
        self.assertEqual(args.command, "replay")
        self.assertEqual(args.tasks, ["task_0003", "task_0004"])

    def test_pareto_parser(self):
        args = self.parser.parse_args(["pareto", "--run", "demo_run"])
        self.assertEqual(args.command, "pareto")
        self.assertEqual(args.source, "both")

    def test_pareto_source_parser(self):
        args = self.parser.parse_args(["pareto", "--run", "demo_run", "--source", "offline_rl"])
        self.assertEqual(args.command, "pareto")
        self.assertEqual(args.source, "offline_rl")

    def test_check_parser(self):
        args = self.parser.parse_args(["check", "--viewer"])
        self.assertEqual(args.command, "check")
        self.assertTrue(args.viewer)

if __name__ == "__main__":
    unittest.main()
