# Trajectory Saving and Loading for Pareto Search

This directory now includes functionality to save and load optimized trajectories from Pareto search experiments, allowing you to visualize individual trajectories post-hoc without re-running the optimization.

## Features

### 1. Trajectory Saving (`pareto_search.py`)
- Automatically saves each successful trajectory during Pareto search
- Creates timestamped experiment directories
- Saves trajectory waypoints, metadata, and cost function values
- Preserves experiment configuration for reproducibility

### 2. Trajectory Loading (`trajectory_loader.py`)
- Load and visualize any saved trajectory individually
- Interactive controls for trajectory replay and analysis
- Preserves original trajectory colors and properties

## Usage

### Running Pareto Search with Trajectory Saving

```bash
# Basic usage (trajectories saved by default)
python pareto_search.py --alpha-start 0.0 --alpha-end 1.0 --alpha-step 0.1

# Custom experiment name
python pareto_search.py --experiment-name "my_experiment" --alpha-step 0.05

# Disable trajectory saving
python pareto_search.py --no-save-trajectories

# With specific parameters
python pareto_search.py \
    --cost-mode sum \
    --alpha-start 0.0 \
    --alpha-end 1.0 \
    --alpha-step 0.1 \
    --experiment-name "detailed_search" \
    --csv-file "my_results.csv"
```

### Loading and Visualizing Saved Trajectories

```bash
# List available experiments
ls src/pareto_data_and_results/

# List trajectories in an experiment
python trajectory_loader.py src/pareto_data_and_results/pareto_search_20240101_123456

# Visualize a specific trajectory
python trajectory_loader.py src/pareto_data_and_results/pareto_search_20240101_123456 --trajectory-id alpha_0p500

# Just list available trajectories
python trajectory_loader.py src/pareto_data_and_results/pareto_search_20240101_123456 --list
```

## Experiment Directory Structure

Each experiment creates a directory with the following structure:

```
src/pareto_data_and_results/pareto_search_YYYYMMDD_HHMMSS/
├── experiment_config.json      # Experiment configuration and parameters
├── trajectory_metadata.json    # Summary of all saved trajectories
├── trajectory_alpha_0p000.pkl  # Individual trajectory files
├── trajectory_alpha_0p100.pkl
├── trajectory_alpha_0p200.pkl
└── ...
```

### File Contents

- **experiment_config.json**: Contains experiment parameters, obstacles, target position, start configuration
- **trajectory_metadata.json**: Summary table of all trajectories with costs and properties
- **trajectory_*.pkl**: Individual trajectory files with waypoints, costs, colors, and metadata

## Interactive Controls

When visualizing a trajectory, you have access to these controls:

- `[r]` - Replay trajectory
- `[c]` - Clear trajectory trace
- `[h]` - Show home position
- `[i]` - Show trajectory information (cost values, weights, etc.)
- `[s]` - Show execution statistics
- `[q]` - Quit demo

## Example Workflow

1. **Run Pareto search:**
   ```bash
   python pareto_search.py --alpha-step 0.05 --experiment-name "fine_search"
   ```

2. **Check what trajectories were saved:**
   ```bash
   python trajectory_loader.py src/pareto_data_and_results/fine_search --list
   ```

3. **Visualize interesting trajectories:**
   ```bash
   # Low length weight (prioritizes obstacle avoidance)
   python trajectory_loader.py src/pareto_data_and_results/fine_search --trajectory-id alpha_0p100
   
   # High length weight (prioritizes short path)
   python trajectory_loader.py src/pareto_data_and_results/fine_search --trajectory-id alpha_0p900
   
   # Balanced weight
   python trajectory_loader.py src/pareto_data_and_results/fine_search --trajectory-id alpha_0p500
   ```

## Benefits

- **Post-hoc analysis**: Examine individual weight combinations without re-running optimization
- **Comparative studies**: Easily compare trajectories with different weight settings
- **Reproducibility**: Exact experiment conditions and results are preserved
- **Efficiency**: No need to re-compute trajectories for visualization
- **Sharing**: Experiment directories can be shared with collaborators

## Notes

- Trajectory files use pickle format for efficient storage of numpy arrays
- Colors are preserved from the original Pareto search plasma colormap
- All experiment parameters are saved for complete reproducibility
- Large experiments may generate many files; consider using descriptive experiment names