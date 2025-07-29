import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Load data
df = pd.read_csv("tradeoff_data_100_w_sum.csv")
lengths = df['length'].values
closenesses = df['closeness'].values
alphas = df['alpha'].values

# Style
colors = plt.cm.plasma(alphas)
marker_size = 60
smaller_marker_size = marker_size * 0.6
label_fontsize = 10
axis_color = '0.4'
axis_linewidth = 2.0

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
ax.set_xlabel("Length", labelpad=8, fontsize=label_fontsize, weight='bold')
ax.set_ylabel("Closeness", labelpad=8, fontsize=label_fontsize, weight='bold')

# Add arrowheads to ends of real axes
arrowstyle = dict(arrowstyle='-|>', color=axis_color, linewidth=axis_linewidth)
arrow_size = 10

# Arrow at end of x-axis
ax.add_patch(FancyArrowPatch((x_max, 0), (x_max + 0.015, 0),
                             **arrowstyle, mutation_scale=arrow_size))

# Arrow at end of y-axis
ax.add_patch(FancyArrowPatch((0, y_max), (0, y_max + 0.015),
                             **arrowstyle, mutation_scale=arrow_size))

# Save and show
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig("Pareto_front_100_w_sum.png", dpi=300)
plt.show()
