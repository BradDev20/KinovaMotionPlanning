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

**See a Rollout in Action (Replay)**
View a task using the interactive viewer:
```bash
python -m src.cli replay --run my_experiment --mode sum --task <task_id> --source both
```
*(The `--source both` flag will show trajectories from both the planner dataset and the offline RL evaluation. You can also specify multiple tasks with `--tasks <id1> <id2>`)*

---

## 2. File Locations

By default, all runs are stored in the `data/runs/` directory. For a given run name (e.g., `my_experiment`), the layout is structured by mode (`sum` or `max`):

- **Root directory:** `data/runs/{run_name}/`
- **Mode directory:** `data/runs/{run_name}/{mode}/` (e.g., `data/runs/my_experiment/sum/`)
  - `dataset/`: Contains the collected planner dataset and `dataset_summary.json`.
  - `checkpoints/{mode}_iql/`: Contains the model checkpoints (e.g., `checkpoint.pt`).
  - `evaluation/`: Contains evaluation rollout results (e.g., `rollouts.pkl`).
  - `pipeline_summary.json`: Summary of the pipeline's execution and support check results.
- **Comparison directory:** `data/runs/{run_name}/compare/` (Generated when evaluating/comparing sum vs. max datasets).

---

## 3. Relevant Command Flags

Here are the key flags used for collection, training, and the pipeline. *(Note: some flags use a slightly different name in the CLI than they do conceptually).*

### Collection & General Flags
- `--run`: The identifier/name for your experiment. Data and results are saved under this name.
- `--mode`: The planner and scalarization mode. Must be either `sum` (weighted sum) or `max` (weighted max).
- `--family`: (Corresponds to **task family**). The task family to sample from (e.g., `mixed`). Defaults to `mixed`.
- `--task-count`: Number of tasks to sample for the dataset.
- `--alpha-count`: (Corresponds to **num alphas**). Number of alpha values (trade-off weights) to sample when an explicit grid is not provided. Default is 11.
- `--alpha-grid`: Comma-separated list of explicit alpha values (e.g., `0.0,0.5,1.0`). Overrides `--alpha-count`.
- `--num-workers`: Number of CPU workers to use for parallel dataset collection.
- `--restart-count`: Planner restarts per task/alpha.

### Training Flags
- `--epochs`: Number of epochs to train the offline RL policy.
- `--batch-size`: Batch size for training (default: 256).
- `--hidden-dim`: Width of the hidden layers (default: 256).
- `--lr`: Learning rate (default: 3e-4).
- `--alpha-conditioning-mode`: How alpha is conditioning during training (`dataset` or `uniform`).

## 4. Task Families

The primary goal of evaluating weighted sum vs. weighted max is to test performance on environments with **convex vs. nonconvex Pareto fronts**. Weighted sum theoretically struggles to find policies on the interior of a nonconvex Pareto front, whereas weighted max can explore these areas effectively.

You can select a specific task family during dataset collection using the `--family` flag, or use `--family mixed` to sample from a mix. 

### Convex Families
These families have simple, convex Pareto fronts (trade-offs between path length, smoothness, and obstacle clearance are straightforward):
- **`stacked_detour`**: Obstacles are stacked vertically, requiring the robot to maneuver clearly over or under them.
- **`asymmetric_safe_margin`**: Obstacles have uneven clearance margins, forcing the robot to prefer one side over another.
- **`corridor_left_right`** *(Legacy)*: A straightforward path flanked symmetrically by obstacles on both the left and right sides.
- **`pinch_point`** *(Legacy)*: A simple, narrow gap between obstacles that the robot must thread through.

### Nonconvex Families
These families are specifically designed to have **nonconvex Pareto fronts**, where the robot must make hard, mutually exclusive choices (e.g., going completely around the left vs. right side of a large obstacle complex):
- **`pinch_bottleneck`**: A restrictive bottleneck created by multiple tight obstacles that drastically limits the viable passage space.
- **`double_corridor`**: Two parallel corridors separated by an obstacle, requiring the robot to commit to one distinct path early on.
- **`culdesac_escape`**: A dead-end or pocket-like arrangement of obstacles that traps the robot if it takes an incorrect approach.
- **`offset_gate`**: A series of staggered gates or openings requiring a zig-zag trajectory to successfully navigate.

---

## 5. Visualization and Graphing

The CLI includes built-in tools for comparing and visualizing the results of the `sum` and `max` modes.

**Compare Sum vs. Max Datasets**
To compare the datasets generated by the two modes and compute size-matched ablation metrics:
```bash
python -m src.cli compare --run my_experiment
```
*Results are saved to `data/runs/my_experiment/compare/`.*

**Plotting Pareto Fronts**
To visualize the Pareto fronts of the planner dataset vs. the evaluation rollouts:
```bash
# Plot Pareto fronts for a specific mode
python -m src.cli pareto --run my_experiment --mode sum

# Plot Pareto comparison between sum and max for the entire run
python -m src.cli pareto --run my_experiment
```
*The resulting Pareto graphs (e.g., `pareto.png`) are saved in the mode's evaluation directory or the run's comparison directory, respectively.*

**Replay Viewer**
The `replay` command allows you to visually inspect the specific trajectories in the environment:
```bash
python -m src.cli replay --run my_experiment --mode sum --tasks <task_id>
```
*Use the `--no-viewer` flag if you only want to process the trajectories without launching the GUI.*
