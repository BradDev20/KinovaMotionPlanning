import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Pareto front from CSV data")
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to the CSV file containing 'length', 'closeness', 'alpha' columns"
    )
    parser.add_argument(
        "--output_folder", type=str, default="pareto_data_and_results",
        help="Folder to save the plot (default: pareto_data_and_results)"
    )
    parser.add_argument(
        "--output_filename", type=str, default=None,
        help="Output plot filename (default: <CSV_basename>.png)"
    )
    parser.add_argument(
        "--cost_mode", type=str, default=None,
        help="Label for objective, e.g., W_sum or W_max. If not provided, will try to infer from CSV name."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Make output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_csv)
    lengths = df['length'].values
    closenesses = df['closeness'].values
    alphas = df['alpha'].values

    # Determine output filename
    if args.output_filename:
        plot_filename = args.output_filename
    else:
        base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        plot_filename = base_name.replace("tradeoff_data", "Pareto_front", 1)

    plot_path = os.path.join(args.output_folder, plot_filename)

    # Plot style
    cmap = plt.cm.plasma
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())
    colors = cmap(norm(alphas))
    marker_size = 25
    small_marker_size = marker_size * 0.6
    label_fontsize = 10
    axis_color = '0.4'
    axis_linewidth = 2.0
    font_family = 'serif'

    # Axis limits
    x_min, x_max = lengths.min() - 0.05, lengths.max() + 0.05
    y_min, y_max = closenesses.min() - 0.05, closenesses.max() + 0.05

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

    # Scatter plot
    ax.scatter(lengths, closenesses,
               c=colors,
               s=small_marker_size,
               marker='D',
               edgecolor='k',
               linewidth=0.6,
               zorder=3)

    # Colorbar for alpha
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Alpha", fontsize=label_fontsize, family=font_family)

    # Title inside plot
    if args.cost_mode:
        objective_label = args.cost_mode
    else:
        # Infer from filename as before
        csv_base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        if "sum" in csv_base_name:
            objective_label = "sum"
        elif "max" in csv_base_name:
            objective_label = "max"
        else:
            objective_label = "W"

    ax.text(0.95, 1.1, f"Trade-offs between objectives for W_{objective_label}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=7,
            family=font_family,
            color='black')

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Style axes
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(axis_linewidth)
        ax.spines[spine].set_color(axis_color)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("Length", labelpad=7, fontsize=label_fontsize,
                  family=font_family)
    ax.set_ylabel("Closeness", labelpad=7, fontsize=label_fontsize,
                  family=font_family)

    # Add arrowheads
    arrow_style = dict(arrowstyle='-|>', color=axis_color, linewidth=axis_linewidth)
    arrow_size = 10
    ax.add_patch(FancyArrowPatch((x_max, 0), (x_max + 0.015, 0),
                                 **arrow_style, mutation_scale=arrow_size))
    ax.add_patch(FancyArrowPatch((0, y_max), (0, y_max + 0.015),
                                 **arrow_style, mutation_scale=arrow_size))

    # Save and show
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()