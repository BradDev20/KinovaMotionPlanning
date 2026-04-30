# Multi-Objective Reinforcement Learning (MORL)

This document provides instructions for running and evaluating the weighted sum vs. weighted max Multi-Objective RL models in this environment. The primary interface for all operations is the `src.cli` module.

## 1. Core Commands

You can run the entire pipeline (collection, training, and evaluation) in one go, or run the steps individually. All commands use the `src.cli` module.

### Run the Full Pipeline
The `pipeline` command runs dataset collection, offline RL training, and evaluation sequentially:
```bash
# Run for weighted sum
python -m src.cli pipeline --run my_experiment --mode sum

# Run for weighted max
python -m src.cli pipeline --run my_experiment --mode max
```

### Individual Steps

**Collect a Dataset**
```bash
python -m src.cli collect --run my_experiment --mode sum --family mixed --task-count 30 --num-workers 4
```

**Train an Offline Policy**
```bash
python -m src.cli train --run my_experiment --mode sum --epochs 50
```

**Evaluate the Trained Policy**
```bash
python -m src.cli eval --run my_experiment --mode sum
```

---

## 2. Reproducing a Comparison Result

To reproduce a result comparing `sum` and `max` modes on a specific non-convex family like `double_corridor` (often abbreviated as `dc`), follow these steps:

### Step 1: Data Collection
Generate datasets for both modes. For a high-quality Pareto front, use a higher task count and multiple restarts.
```bash
# Weighted Sum collection
python -m src.cli collect --run pareto_dc --mode sum --family double_corridor --task-count 50 --restart-count 5

# Weighted Max collection
python -m src.cli collect --run pareto_dc --mode max --family double_corridor --task-count 50 --restart-count 5
```

### Step 2: Training
Train the IQL agents on the collected data. 50-100 epochs is usually sufficient for convergence.
```bash
python -m src.cli train --run pareto_dc --mode sum --epochs 100
python -m src.cli train --run pareto_dc --mode max --epochs 100
```

### Step 3: Evaluation
Run rollouts using the trained policies across a range of alpha (preference) values.
```bash
python -m src.cli eval --run pareto_dc --mode sum --alpha-grid 0.0,0.25,0.5,0.75,1.0
python -m src.cli eval --run pareto_dc --mode max --alpha-grid 0.0,0.25,0.5,0.75,1.0
```

---

## 3. Visualization and Graphing

Once you have evaluation results, you can generate graphs to analyze the performance.

### Pareto Front Comparison
To see how well each mode covers the objective space (Path Length vs. Obstacle Distance):
```bash
# Comparison graph for both modes
python -m src.cli pareto --run pareto_dc
```
*Graph saved to: `data/runs/pareto_dc/compare/pareto.png`*

### Trajectory Replay (Visual Inspection)
To visually compare the paths taken by the `sum` agent vs the `max` agent for a specific task:
```bash
python -m src.cli replay --run pareto_dc --mode max --task task_0001 --source both
```
- `--source both`: Shows the planner's original path and the RL agent's rollout.
- `--no-viewer`: Use this flag to just print stats if you don't have a display.

### Detailed Dataset Comparison
To compute hypervolume, success rates, and coverage metrics across the two modes:
```bash
python -m src.cli compare --run pareto_dc
```
*Summary saved to: `data/runs/pareto_dc/compare/comparison_summary.json`*

---

## 4. File Locations

All results are stored in the `data/runs/` directory.

- **Mode directory:** `data/runs/{run_name}/{mode}/`
  - `dataset/`: Collected planner trajectories and `dataset_summary.json`.
  - `checkpoints/`: Saved IQL model (`checkpoint.pt`).
  - `evaluation/`: Rollout data (`rollouts.pkl`) and metrics.
- **Comparison directory:** `data/runs/{run_name}/compare/`
  - `pareto.png`: The objective-space visualization.
  - `comparison_summary.json`: High-level metrics (Hypervolume, etc.)

---

## 5. Detailed Command Line Flag List

The CLI supports several subcommands (`collect`, `train`, `eval`, `pipeline`, `compare`, `replay`, `pareto`). Below are the most important flags grouped by their function.

### Common Flags (Used across most commands)
- `--run`: Identifier for the experiment. Controls default file paths.
- `--mode`: Either `sum` or `max`. Sets the scalarization/optimization strategy.
- `--root`: Root directory for all data/runs (defaults to `data/runs`).
- `--device`: Torch device (`cpu`, `cuda`, `cuda:0`, etc.).

### Collection Flags (`collect` and `pipeline`)
- `--family`: Task family to sample from (e.g., `mixed`, `double_corridor`, `culdesac_escape`).
- `--regime`: Geometry regime (`mixed`, `convex`, `nonconvex`).
- `--difficulty`: Difficulty level (`easy`, `medium`, `hard`).
- `--task-count`: Total number of unique scenes to sample.
- `--alpha-count`: Number of weight trade-offs to sample per task (default: 11).
- `--alpha-grid`: Explicit comma-separated list of alpha values (overrides `--alpha-count`).
- `--restart-count`: Number of times the planner restarts per task/alpha to find a better path.
- `--num-workers`: Number of parallel CPU workers for task processing.
- `--gpu-batch-size`: Batch size for the Torch-based trajectory optimizer.
- `--planner-steps`: Number of optimization steps per trajectory (default: 500).
- `--seed-bank-dir`: Directory for global "warm-start" seeds (defaults to `seed_bank`). "Warm-start" seeds are faster becauase they are known to be successful.
- `--report-size-matched`: Flag to enable size-matched max ablation reporting in comparisons.

### Training Flags (`train` and `pipeline`)
- `--epochs`: Number of training epochs (default: 50).
- `--batch-size`: Mini-batch size for IQL training (default: 256).
- `--lr`: Learning rate for the Adam optimizer (default: 1e-4).
- `--hidden-dim`: Width of the MLP layers in the Q and Policy networks (default: 256).
- `--alpha-conditioning-mode`: How preferences are sampled during training (`dataset` or `uniform`).
- `--expectile`: The IQL expectile parameter (default: 0.7).
- `--beta`: Inverse temperature for the advantage-weighted policy extraction (default: 3.0).

### Evaluation Flags (`eval` and `pipeline`)
- `--split`: Which dataset split to evaluate (`train`, `val`, `test`).
- `--alpha-grid`: Comma-separated weights to test during evaluation rollouts.
- `--deterministic`: If set, use the policy mean instead of sampling actions.
- `--max-steps`: Maximum horizon for the RL rollouts (overrides environment default).

### Visualization & Analysis Flags
- **`replay`**:
  - `--task`: Single task ID to replay.
  - `--source`: Where to pull trajectories from (`both`, `planner`, or `eval`).
  - `--no-viewer`: Process data and print stats without opening the 3D GUI.
- **`pareto`**:
  - `--group-by-family`: Create a grid of plots, one for each task family.
  - `--coverage-only`: Plot only the points on the Pareto front.
  - `--nonconvex-only`: Only plot families designated as non-convex.
- **`compare`**:
  - `--size-matched-samples`: Number of bootstrap samples for the max-ablation comparison.
