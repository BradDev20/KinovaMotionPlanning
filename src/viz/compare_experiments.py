#!/usr/bin/env python3
"""
Multi-Experiment Pareto Front Comparison

Plots Pareto fronts from multiple experiments on the same figure.
Each experiment gets a different color, allowing comparison of different
optimization formulations (e.g., sum vs max modes) or parameter settings.
"""

import sys
import os
import numpy as np
import argparse
import json
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../examples'))

from trajectory_loader import TrajectoryLoader

# EXPERIMENT DIRECTORIES TO COMPARE
EXPERIMENT_DIRECTORIES = [
    # "/Users/connor/Desktop/DevProjects/KinovaMotionPlanning/src/pareto_data_and_results/toy_sum",
    # "/Users/connor/Desktop/DevProjects/KinovaMotionPlanning/src/pareto_data_and_results/toy_max",
    "/Users/connor/Desktop/DevProjects/KinovaMotionPlanning/src/pareto_data_and_results/quick_test"
]

# Color palette for experiments
EXPERIMENT_COLORS = [
    '#2E86AB',  # Blue
    '#A23B72',  # Purple/Pink  
    '#F18F01',  # Orange
    '#C73E1D',  # Red
    '#6A994E',  # Green
    '#577590',  # Blue-gray
    '#F8961E',  # Yellow-orange
    '#43AA8B',  # Teal
]


class ExperimentData:
    """Container for experiment data and metadata"""
    
    def __init__(self, experiment_dir: str):
        """Load experiment data"""
        self.experiment_dir = experiment_dir
        self.experiment_name = os.path.basename(experiment_dir)
        
        try:
            self.loader = TrajectoryLoader(experiment_dir)
            self.config = self.loader.config
            self.metadata = self.loader.metadata
            
            # Extract cost data
            self.alphas = np.array([t['alpha'] for t in self.metadata])
            self.length_costs = np.array([t['length_cost'] for t in self.metadata])
            self.obstacle_costs = np.array([t['obstacle_cost'] for t in self.metadata])
            
            # Initialize dominated mask (all non-dominated by default)
            self.is_dominated = np.zeros(len(self.metadata), dtype=bool)
            
            self.loaded_successfully = True
            print(f"✓ Loaded {len(self.metadata)} trajectories from {self.experiment_name}")
            
        except Exception as e:
            print(f"✗ Failed to load {experiment_dir}: {e}")
            self.loaded_successfully = False
            self.alphas = np.array([])
            self.length_costs = np.array([])
            self.obstacle_costs = np.array([])
            self.is_dominated = np.array([], dtype=bool)
    
    def get_display_name(self) -> str:
        """Generate a nice display name for the experiment"""
        if hasattr(self, 'config') and 'cost_mode' in self.config:
            cost_mode = self.config['cost_mode'].upper()
            return f"{self.experiment_name} ({cost_mode})"
        return self.experiment_name
    
    def get_metadata_info(self) -> Dict:
        """Get summary information about the experiment"""
        if not self.loaded_successfully:
            return {}
        
        return {
            'cost_mode': self.config.get('cost_mode', 'unknown'),
            'alpha_range': (self.config.get('alpha_start', 0), self.config.get('alpha_end', 1)),
            'alpha_step': self.config.get('alpha_step', 0.1),
            'num_trajectories': len(self.metadata),
            'length_range': (self.length_costs.min(), self.length_costs.max()),
            'obstacle_range': (self.obstacle_costs.min(), self.obstacle_costs.max())
        }
    
    def identify_dominated_solutions(self):
        """
        Identify Pareto dominated solutions within this experiment.
        A solution is dominated if there exists another solution that is
        better in at least one objective and not worse in any objective.
        """
        n = len(self.length_costs)
        self.is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Check if j dominates i (both objectives are being minimized)
                length_better = self.length_costs[j] <= self.length_costs[i]
                obstacle_better = self.obstacle_costs[j] <= self.obstacle_costs[i]
                strictly_better = (self.length_costs[j] < self.length_costs[i] or 
                                  self.obstacle_costs[j] < self.obstacle_costs[i])
                
                if length_better and obstacle_better and strictly_better:
                    self.is_dominated[i] = True
                    break
        
        num_dominated = np.sum(self.is_dominated)
        num_pareto = n - num_dominated
        print(f"  → {self.experiment_name}: {num_pareto} Pareto optimal, {num_dominated} dominated")


class MultiExperimentPlotter:
    """Plots multiple experiments on the same Pareto front"""
    
    def __init__(self, experiment_dirs: List[str], filter_dominated: bool = False):
        """Initialize with list of experiment directories"""
        self.experiments = []
        self.filter_dominated = filter_dominated
        
        # Load all experiments
        for exp_dir in experiment_dirs:
            if os.path.exists(exp_dir):
                exp_data = ExperimentData(exp_dir)
                if exp_data.loaded_successfully:
                    self.experiments.append(exp_data)
            else:
                print(f"Warning: Experiment directory not found: {exp_dir}")
        
        if not self.experiments:
            raise ValueError("No valid experiments found!")
        
        print(f"\nLoaded {len(self.experiments)} experiments for comparison")
        
        # Identify dominated solutions if filtering is enabled
        if self.filter_dominated:
            print("\nIdentifying dominated solutions...")
            for exp in self.experiments:
                exp.identify_dominated_solutions()
    
    def create_comparison_plot(self, figsize: Tuple[float, float] = (6, 4)) -> plt.Figure:
        """Create the multi-experiment comparison plot"""
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Calculate overall bounds for consistent scaling
        all_length_costs = np.concatenate([exp.length_costs for exp in self.experiments])
        all_obstacle_costs = np.concatenate([exp.obstacle_costs for exp in self.experiments])
        
        # Plot each experiment
        legend_handles = []
        legend_labels = []
        
        # Determine if we should use mode-based coloring (only for single experiments)
        use_mode_coloring = len(self.experiments) == 1
        
        for i, experiment in enumerate(self.experiments):
            # Assign color based on experiment mode if only one experiment
            if use_mode_coloring:
                cost_mode = experiment.config.get('cost_mode', 'unknown')
                if cost_mode == 'sum':
                    color = '#2E86AB'  # Blue
                elif cost_mode == 'max':
                    color = '#A23B72'  # Purple
                elif cost_mode == 'max_constrained':
                    color = '#F18F01'  # Orange
                else:
                    color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            else:
                color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            
            if self.filter_dominated:
                # Separate Pareto optimal and dominated solutions
                pareto_mask = ~experiment.is_dominated
                dominated_mask = experiment.is_dominated
                
                # Plot Pareto optimal solutions
                if np.any(pareto_mask):
                    scatter = ax.scatter(
                        experiment.length_costs[pareto_mask], 
                        experiment.obstacle_costs[pareto_mask],
                        c=color,
                        s=50,
                        marker='D',
                        edgecolor='white',
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=3,
                        label=experiment.get_display_name()
                    )
                    legend_handles.append(scatter)
                    legend_labels.append(experiment.get_display_name())
                
                # Plot dominated solutions with 'x' markers (no legend)
                if np.any(dominated_mask):
                    ax.scatter(
                        experiment.length_costs[dominated_mask], 
                        experiment.obstacle_costs[dominated_mask],
                        c=color,
                        s=40,
                        marker='x',
                        linewidth=1.5,
                        alpha=0.4,
                        zorder=2
                    )
            else:
                # Plot all points normally
                scatter = ax.scatter(
                    experiment.length_costs, 
                    experiment.obstacle_costs,
                    c=color,
                    s=50,
                    marker='D',
                    edgecolor='white',
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=3,
                    label=experiment.get_display_name()
                )
                legend_handles.append(scatter)
                legend_labels.append(experiment.get_display_name())
        
        # Style the plot
        self._style_axes(ax, all_length_costs, all_obstacle_costs)
        
        # Add legend
        ax.legend(
            legend_handles, 
            legend_labels,
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
            title='Experiments'
        )
        
        # Add title
        title = 'Pareto Front Comparison Across Experiments'
        if self.filter_dominated:
            title += '\n(Dominated solutions marked with ×)'
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        return fig
    
    def _style_axes(self, ax, all_length_costs: np.ndarray, all_obstacle_costs: np.ndarray):
        """Apply styling to the axes"""
        # Calculate axis limits
        x_margin = (all_length_costs.max() - all_length_costs.min()) * 0.1
        y_margin = (all_obstacle_costs.max() - all_obstacle_costs.min()) * 0.1
        
        x_min = all_length_costs.min() - x_margin
        x_max = all_length_costs.max() + x_margin
        y_min = all_obstacle_costs.min() - y_margin  
        y_max = all_obstacle_costs.max() + y_margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Style spines
        axis_color = '0.4'
        axis_linewidth = 2.0
        
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(axis_linewidth)
            ax.spines[spine].set_color(axis_color)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Add ticks and labels
        x_tick_values = np.linspace(all_length_costs.min(), all_length_costs.max(), 5)
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels([f'{val:.1f}' for val in x_tick_values], fontsize=10, family='serif')
        
        y_tick_values = np.linspace(all_obstacle_costs.min(), all_obstacle_costs.max(), 5)
        ax.set_yticks(y_tick_values)
        ax.set_yticklabels([f'{val:.1f}' for val in y_tick_values], fontsize=10, family='serif')
        
        # Labels
        ax.set_xlabel("Length Cost", labelpad=10, fontsize=12, family='serif', fontweight='bold')
        ax.set_ylabel("Obstacle Cost", labelpad=10, fontsize=12, family='serif', fontweight='bold')
        
        # Add arrow heads
        arrow_style = dict(arrowstyle='-|>', color=axis_color, linewidth=axis_linewidth)
        arrow_size = 12
        ax.add_patch(FancyArrowPatch((x_max, y_min), (x_max + x_margin*0.2, y_min),
                                    **arrow_style, mutation_scale=arrow_size))
        ax.add_patch(FancyArrowPatch((x_min, y_max), (x_min, y_max + y_margin*0.2),
                                    **arrow_style, mutation_scale=arrow_size))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    def print_experiment_summary(self):
        """Print summary information about all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*80)
        
        for i, exp in enumerate(self.experiments):
            info = exp.get_metadata_info()
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            
            print(f"\n{exp.get_display_name()}")
            print(f"  Directory: {exp.experiment_dir}")
            print(f"  Color: {color}")
            print(f"  Cost Mode: {info.get('cost_mode', 'unknown')}")
            print(f"  Alpha Range: {info.get('alpha_range', (0, 1))} (step: {info.get('alpha_step', 'unknown')})")
            print(f"  Trajectories: {info.get('num_trajectories', 0)}")
            print(f"  Length Cost Range: [{info.get('length_range', (0, 0))[0]:.2f}, {info.get('length_range', (0, 0))[1]:.2f}]")
            print(f"  Obstacle Cost Range: [{info.get('obstacle_range', (0, 0))[0]:.2f}, {info.get('obstacle_range', (0, 0))[1]:.2f}]")
        
        print("="*80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare Pareto fronts from multiple experiments')
    
    parser.add_argument('--experiments', nargs='+', type=str,
                        help='List of experiment directories to compare (overrides default list)')
    parser.add_argument('--output', type=str, default='pareto_comparison.png',
                        help='Output plot filename (default: pareto_comparison.png)')
    parser.add_argument('--output-dir', type=str, default='src/viz',
                        help='Output directory (default: src/viz)')
    parser.add_argument('--width', type=float, default=6,
                        help='Figure width in inches (default: 6)')
    parser.add_argument('--height', type=float, default=4,
                        help='Figure height in inches (default: 4)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    parser.add_argument('--summary', action='store_true',
                        help='Print detailed experiment summary')
    parser.add_argument('--filter-dominated', action='store_true',
                        help='Filter out Pareto dominated solutions (mark with x)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Determine which experiments to use
    experiment_dirs = args.experiments if args.experiments else EXPERIMENT_DIRECTORIES
    
    # Check if experiment directories exist
    missing_dirs = [d for d in experiment_dirs if not os.path.exists(d)]
    if missing_dirs:
        print("Warning: The following experiment directories were not found:")
        for d in missing_dirs:
            print(f"  - {d}")
        experiment_dirs = [d for d in experiment_dirs if os.path.exists(d)]
    
    if not experiment_dirs:
        print("Error: No valid experiment directories found!")
        return
    
    try:
        # Create plotter and generate comparison
        print("Loading experiments for comparison...")
        plotter = MultiExperimentPlotter(experiment_dirs, filter_dominated=args.filter_dominated)
        
        if args.summary:
            plotter.print_experiment_summary()
        
        # Create plot
        print(f"\nCreating comparison plot...")
        fig = plotter.create_comparison_plot(figsize=(args.width, args.height))
        
        # Save plot
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output)
        
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"Comparison plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 