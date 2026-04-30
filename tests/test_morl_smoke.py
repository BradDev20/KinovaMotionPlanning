import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("MORL_FULL_SMOKE") == "1", "Set MORL_FULL_SMOKE=1 to run the full planner smoke test.")
class MORLSmokeTests(unittest.TestCase):
    def test_end_to_end_cli_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "data"
            collect_cmd = [
                sys.executable,
                "-m",
                "src.morl.collect_dataset",
                "--experiment-name",
                "smoke",
                "--task-count",
                "2",
                "--alpha-grid",
                "0.0,1.0",
                "--planner-mode",
                "sum",
                "--restart-count",
                "1",
                "--seed",
                "1",
                "--output-root",
                str(output_root),
                "--device",
                "cpu",
                "--num-workers",
                "1",
                "--n-waypoints",
                "8",
                "--planner-steps",
                "8",
                "--planner-max-iter",
                "15",
                "--planner-max-fun",
                "60",
                "--repair-max-iter",
                "8",
                "--repair-max-fun",
                "32",
                "--max-steps",
                "8",
            ]
            subprocess.check_call(collect_cmd, cwd=repo_root)


if __name__ == "__main__":
    unittest.main()
