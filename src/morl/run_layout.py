from __future__ import annotations

from pathlib import Path

MODES = frozenset({"sum", "max"})
DEFAULT_RUNS_ROOT = Path("data/runs")
DEFAULT_LEGACY_ROOT = Path("data/morl")


def mode_root_for_run(run_name: str, mode: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Helper to find the root folder for a specific mode (sum/max) within a run."""
    return Path(root) / str(run_name) / str(mode)


def dataset_dir_for_run(run_name: str, mode: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Finds where the collected trajectories are stored for a run/mode."""
    return mode_root_for_run(run_name, mode, root=root) / "dataset"


def checkpoints_root_for_run(run_name: str, mode: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Base folder for all model checkpoints in a run."""
    return mode_root_for_run(run_name, mode, root=root) / "checkpoints"


def checkpoint_dir_for_run(run_name: str, mode: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Specific folder for the current mode's model checkpoints."""
    return checkpoints_root_for_run(run_name, mode, root=root) / f"{mode}_iql"


def evaluation_dir_for_run(run_name: str, mode: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Where we dump the rollout results and metrics from evaluation."""
    return mode_root_for_run(run_name, mode, root=root) / "evaluation"


def pipeline_summary_path_for_run(run_name: str, mode: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """The JSON file that summarizes how the whole pipeline run went."""
    return mode_root_for_run(run_name, mode, root=root) / "pipeline_summary.json"


def compare_dir_for_run(run_name: str, root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Folder for comparison graphs and reports between sum and max."""
    return Path(root) / str(run_name) / "compare"


def is_flattened_dataset_dir(path: str | Path) -> bool:
    """Checks if a directory follows our standard 'dataset' folder naming convention."""
    candidate = Path(path)
    return candidate.name == "dataset" and candidate.parent.name in MODES


def default_training_output_dir(dataset_dir: str | Path, scalarizer: str) -> Path:
    """Guesses where to save trained models based on where the dataset is."""
    candidate = Path(dataset_dir)
    if is_flattened_dataset_dir(candidate):
        return candidate.parent / "checkpoints" / f"{scalarizer}_iql"
    return candidate / "checkpoints" / f"{scalarizer}_iql"


def infer_dataset_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    """Tries to work backwards from a model file to find the dataset it was trained on."""
    checkpoint = Path(checkpoint_path)
    if checkpoint.name == "checkpoint.pt" and checkpoint.parent.parent.name == "checkpoints":
        flattened_candidate = checkpoint.parent.parent.parent / "dataset"
        if flattened_candidate.exists():
            return flattened_candidate
    return checkpoint.parent.parent.parent


def default_evaluation_output_dir(checkpoint_path: str | Path, dataset_dir: str | Path | None = None) -> Path:
    """Decides where to put evaluation results."""
    dataset_candidate = Path(dataset_dir) if dataset_dir is not None else infer_dataset_dir_from_checkpoint(checkpoint_path)
    if is_flattened_dataset_dir(dataset_candidate):
        return dataset_candidate.parent / "evaluation"
    return Path(checkpoint_path).parent / "evaluation"


def default_compare_output_dir(sum_dir: str | Path, max_dir: str | Path) -> Path:
    """Figures out a good place to put a comparison report between two datasets."""
    sum_path = Path(sum_dir)
    max_path = Path(max_dir)
    if is_flattened_dataset_dir(sum_path) and is_flattened_dataset_dir(max_path) and sum_path.parent.parent == max_path.parent.parent:
        return sum_path.parent.parent / "compare"
    return sum_path.parent / "benchmark_report"
