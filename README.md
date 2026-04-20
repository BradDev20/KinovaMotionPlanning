# Motion Planning for Kinova Gen3 in MuJoCo

Authors: Connor Mattson, Zohre Karimi, Atharv Belsare

## Quickstart

Run commands from the repo root with your Python environment already activated.

The documented `python -m src.cli ...` commands are shell-neutral and intended to work from bash/zsh on macOS and Linux as well as from PowerShell on Windows.

Install dependencies:

```bash
pip install -r requirements.txt
```

Check that MuJoCo and the robot model load:

```bash
python -m src.cli check
```

Launch the Gen3 viewer sanity check:

```bash
python -m src.cli check --viewer
```

Legacy demos and ad hoc visualization scripts have been archived under `src/legacy/`.

## Unified CLI

The primary user-facing entrypoint is:

```bash
python -m src.cli --help
```

Main workflows:

```bash
python -m src.cli collect --run bench_culdesac_t30_a11_r5_20260413 --mode max --family culdesac_escape
python -m src.cli train --run bench_culdesac_t30_a11_r5_20260413 --mode max
python -m src.cli eval --run bench_culdesac_t30_a11_r5_20260413 --mode max
python -m src.cli replay --run bench_culdesac_t30_a11_r5_20260413 --mode max --task task_0003 --alpha 0.5
python -m src.cli compare --run bench_culdesac_t30_a11_r5_20260413
python -m src.cli pareto --run bench_culdesac_t30_a11_r5_20260413
```

## Run Layout

New runs default to a flattened layout under `data/runs`:

```text
data/runs/<run-name>/
  compare/
  max/
    dataset/
    checkpoints/
    evaluation/
  sum/
    dataset/
    checkpoints/
    evaluation/
```

Examples:

```text
data/runs/bench_culdesac_t30_a11_r5_20260413/max/dataset
data/runs/bench_culdesac_t30_a11_r5_20260413/max/checkpoints/max_iql
data/runs/bench_culdesac_t30_a11_r5_20260413/max/evaluation
data/runs/bench_culdesac_t30_a11_r5_20260413/compare
```

Legacy runs under `data/morl/...` are still supported when you pass explicit paths.

## Common Commands

Collect a legal MORL dataset:

```bash
python -m src.cli collect --run bench_nonconvex_t10_a7_r4_20260413 --mode max --family mixed --regime nonconvex --task-count 10 --alpha-count 7 --restart-count 4
```

Train from the standard flattened dataset path:

```bash
python -m src.cli train --run bench_nonconvex_t10_a7_r4_20260413 --mode max --epochs 200 --device cpu
```

Evaluate the trained checkpoint:

```bash
python -m src.cli eval --run bench_nonconvex_t10_a7_r4_20260413 --mode max --device cpu --deterministic
```

Replay planner and policy trajectories together:

```bash
python -m src.cli replay --run bench_nonconvex_t10_a7_r4_20260413 --mode max --task task_0003 --alpha 0.5
```

Replay only a legacy rollout file:

```bash
python -m src.cli replay --task task_0003 --evaluation-rollouts data/morl/some_legacy_run/max/checkpoints/max_iql/evaluation/rollouts.pkl
```

Generate a Pareto plot:

```bash
python -m src.cli pareto --run bench_nonconvex_t10_a7_r4_20260413 --group-by-family
```

## Notes

- Commands above assume the current working directory is the repo root.
- Paths are resolved with `pathlib` in the CLI and run-layout helpers, so Mac/Linux users do not need any Windows-specific wrappers.
- `python -m src.cli train` and `eval` require PyTorch in the active environment.
- Existing artifact semantics checks still apply. Legacy pre-fix datasets and checkpoints must be rebuilt before reuse in MORL training or evaluation.
- The unified CLI no longer launches archived demo scripts from `check`; use `python -m src.cli check` only for environment/model validation.
