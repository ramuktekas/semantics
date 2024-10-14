import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("Transfer_entropy_task.csv")

# Extract the first value from the array-like strings in the relevant columns
for col in ['te_acw_forward', 'te_raw_forward', 'te_acw_reverse']:
    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Set up the plotting structure (3 rows, 2 columns)
fig, axs = plt.subplots(3, 2, figsize=(8, 10))  # Reduced overall plot height

# Define ROIs and conditions
rois = ['A1', 'TA2', 'PSL']
conditions = ['CS', 'WD']

# Set the x-tick labels
x_labels = ['ACW Forward', 'Raw Forward', 'ACW Reverse']

# Define colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Use distinct colors: blue, orange, green
line_color = '#d62728'  # Color for the y=0 line

# Plotting
bar_width = 0.3  # Set thinner bars
for i, roi in enumerate(rois):
    for j, condition in enumerate(conditions):
        # Filter data for the specific ROI, condition, and group 'All'
        filtered_data = df[(df['ROI'] == roi) & (df['Input'] == condition) & (df['Group'] == 'All')]

        # Check if there's any data for the filtered conditions
        if filtered_data.empty:
            continue  # Skip if no data is available

        # Extract TE values
        te_acw_forward = filtered_data['te_acw_forward'].values[0]  # All 10 values
        te_raw_forward = filtered_data['te_raw_forward'].values[0]  # All 10 values
        te_acw_reverse = filtered_data['te_acw_reverse'].values[0]  # All 10 values

        # Create bar positions for each value
        x = np.arange(len(te_acw_forward))  # 10 values

        # Create bar plots with spacing
        axs[i, j].bar(x - bar_width, te_acw_forward, width=bar_width, color=colors[0], label='ACW Forward')
        axs[i, j].bar(x, te_raw_forward, width=bar_width, color=colors[1], label='Raw Forward')
        axs[i, j].bar(x + bar_width, te_acw_reverse, width=bar_width, color=colors[2], label='ACW Reverse')

        # Set y-axis limits
        axs[i, j].set_ylim(-1, 2.1)

        # Add horizontal grid
        axs[i, j].yaxis.grid(True, linestyle='--', alpha=0.5)

        # Add solid line for y=0
        axs[i, j].axhline(0, color=line_color, linewidth=1.5, linestyle='-')

        # Set titles based on conditions
        title = f"Transfer Entropy ({'Cosine Similarity' if condition == 'CS' else 'Word Depth'} to {roi})"
        axs[i, j].set_title(title)

        # Set x-ticks only for the bottom row plots
        if i == 2:  # Bottom row
            axs[i, j].set_xticks(x)
            axs[i, j].set_xticklabels(range(1, 11))  # Set x-tick labels from 1 to 10
            axs[i, j].set_xlabel("Lags (1 lag = 1 second)")  # X-axis title for bottom row
        else:  # Other rows
            axs[i, j].set_xticks([])  # Remove x-ticks for top rows

# Set a common legend at the bottom
handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles from one of the plots
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit legend

# Save the figure
plt.savefig('TE_Bar_Plots.png', bbox_inches='tight')  # Save with tight bounding box
plt.show()
