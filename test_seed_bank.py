"""Verify seeds save even when CollectionSupportError is raised."""
import sys, json
from pathlib import Path
import numpy as np

DATASET_DIR = "data/runs/seed_bank_partial/max/dataset"
SEED_BANK_DIR = Path(DATASET_DIR) / "seed_bank"

sys.argv = [
    "collect_dataset",
    "--experiment-name", "seed_bank_partial",
    "--planner-mode", "max",
    "--dataset-dir", DATASET_DIR,
    "--task-count", "5",  # too few to pass support check
    "--restart-count", "4",
    "--seed", "0",
    "--output-root", "data/runs",
    "--task-family", "corridor_left_right",
    "--difficulty", "medium",
    "--benchmark-profile", "baseline",
    "--geometry-regime", "mixed",
    "--n-waypoints", "50",
    "--cost-sample-rate", "5",
    "--max-steps", "200",
    "--device", "cuda",
    "--num-workers", "1",
    "--alpha-count", "5",
    "--objective-tol", "0.01",
    "--path-tol", "0.20",
    "--route-tol", "0.20",
    "--alpha-schedule", "linear",
    "--quiet",
]

from src.morl.collect_dataset import main

try:
    main()
    print("MARKER result=ok (no error)", flush=True)
except Exception as e:
    print(f"MARKER result={type(e).__name__}", flush=True)

if SEED_BANK_DIR.exists():
    manifest = json.loads((SEED_BANK_DIR / "manifest.json").read_text())
    total = sum(len(v) for v in manifest.values())
    print(f"MARKER seed_bank_exists=True total_seeds={total} families={list(manifest.keys())}", flush=True)
else:
    print("MARKER seed_bank_exists=False", flush=True)
