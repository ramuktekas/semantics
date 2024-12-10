import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('TE_182subs.csv')

# Extract the first value from the array-like strings in the relevant columns
for col in ['te_acw_forward', 'te_raw_forward']:
    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Set up the plotting structure (2 rows, 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows for TE types, 3 columns for ROIs

# Define ROIs and TE types
rois = ['A1', 'TA2', 'PSL']
te_types = ['ACW Forward', 'Raw Forward']
conditions = ['CS', 'WD']

# Define bright colors for CS and WD
colors = [(0/255, 0/255, 245/255), (55/255, 126/255, 34/255)]  # Bright Blue for CS, Bright Green for WD
line_color = '#d62728'  # Color for the y=0 line

# Plotting
bar_width = 0.3  # Set thinner bars
for i, te_type in enumerate(te_types):  # Iterate through TE types (ACW Forward, Raw Forward)
    for j, roi in enumerate(rois):  # Iterate through ROIs (A1, TA2, PSL)
        # Filter data for the specific ROI and TE type, and group 'All'
        filtered_data = df[(df['ROI'] == roi) & (df['Group'] == 'All')]

        # Check if there's any data for the filtered conditions
        if filtered_data.empty:
            continue  # Skip if no data is available

        # Extract TE values for the current TE type for both CS and WD
        te_values_cs = filtered_data[filtered_data['Input'] == 'CS'][f'te_{te_type.lower().replace(" ", "_")}'].values[0]
        te_values_wd = filtered_data[filtered_data['Input'] == 'WD'][f'te_{te_type.lower().replace(" ", "_")}'].values[0]

        # Create bar positions for each value
        x = np.arange(len(te_values_cs))  # 10 values for each lag

        # Plot bar plots for each lag, with separate bars for CS and WD
        axs[i, j].bar(x - bar_width/2, te_values_cs, width=bar_width, color=colors[0], label='Cosine Similarity')
        axs[i, j].bar(x + bar_width/2, te_values_wd, width=bar_width, color=colors[1], label='Word Depth')

        # Set y-axis limits
        axs[i, j].set_ylim(0, 4.3)

        # Add horizontal grid
        axs[i, j].yaxis.grid(True, linestyle='--', alpha=0.5)

        # Add solid line for y=0
        axs[i, j].axhline(0, color=line_color, linewidth=1.5, linestyle='-')

        # Set titles based on TE type and ROI, making the titles bold
        if te_type == 'ACW Forward':
            axs[i, j].set_title(f'Transfer Entropy Dynamic ACW-0 {roi}', fontweight='bold')
        else:
            axs[i, j].set_title(f'Transfer Entropy BOLD signal {roi}', fontweight='bold')

        # Set x-ticks and labels
        axs[i, j].set_xticks(x)
        axs[i, j].set_xticklabels(range(1, 11))  # Set x-tick labels from 1 to 10

        # Set x-axis title only for the bottom-most plots (last row)
        if i == 1:  # Bottom row
            axs[i, j].set_xlabel("Lags (1 second)", fontweight='bold')

        # Set y-axis label only for the left-most plots (first column)
        if j == 0:  # Left-most column
            axs[i, j].set_ylabel("Transfer Entropy", fontweight='bold')

# Set a common legend at the bottom
handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles from one of the plots
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit legend

# Save the figure
plt.savefig('TE_Bar_Plots182subs.png', bbox_inches='tight')  # Save with tight bounding box
plt.show()

