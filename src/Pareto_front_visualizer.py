import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Define folder name
folder_name = "pareto_data_and_results"

# Create folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# File paths
data_path = os.path.join(folder_name, "tradeoff_data_100_sphere_max.csv")
plot_path = os.path.join(folder_name, "Pareto_front_100_sphere_max.png")

# Load data
df = pd.read_csv(data_path)
lengths = df['length'].values
closenesses = df['closeness'].values
alphas = df['alpha'].values

# Style
cmap = plt.cm.plasma
norm = Normalize(vmin=min(alphas), vmax=max(alphas))
colors = cmap(norm(alphas))
marker_size = 25
smaller_marker_size = marker_size * 0.6
label_fontsize = 10
axis_color = '0.4'
axis_linewidth = 2.0
font = 'serif'

# Limits
x_min, x_max = lengths.min() - 0.05, lengths.max() + 0.05
y_min, y_max = closenesses.min() - 0.05, closenesses.max() + 0.05

# Create plot
fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

# Scatter points
for i in range(len(alphas)):
    ax.scatter(lengths[i], closenesses[i],
               color=colors[i],
               s=smaller_marker_size,
               marker='D',
               edgecolor='k',
               linewidth=0.6,
               zorder=3)

# Add colorbar for alpha
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy for colorbar
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Alpha", fontsize=label_fontsize, family=font)

# Add fixed-position label inside plot frame (top-right corner)
ax.text(0.95, 1.1, "Trade-offs between objectives for W_max",
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=7,
        weight='normal',
        family=font,
        color='black')

# Set limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Make left and bottom axes visible, thick, and gray
for spine in ['left', 'bottom']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(axis_linewidth)
    ax.spines[spine].set_color(axis_color)

# Hide other spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Hide tick labels but keep axes
ax.set_xticks([])
ax.set_yticks([])

# Axis labels
ax.set_xlabel("Length", labelpad=7, fontsize=label_fontsize, family=font, weight='normal')
ax.set_ylabel("Closeness", labelpad=7, fontsize=label_fontsize, family=font, weight='normal')

# Add arrowheads to ends of real axes
arrowstyle = dict(arrowstyle='-|>', color=axis_color, linewidth=axis_linewidth)
arrow_size = 10

ax.add_patch(FancyArrowPatch((x_max, 0), (x_max + 0.015, 0),
                             **arrowstyle, mutation_scale=arrow_size))
ax.add_patch(FancyArrowPatch((0, y_max), (0, y_max + 0.015),
                             **arrowstyle, mutation_scale=arrow_size))

# Save and show
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig(plot_path, dpi=300)
plt.show()
