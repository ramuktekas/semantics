import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'Transfer_entropy_task.csv'  # Replace with your file path
te_data = pd.read_csv(file_path)

# Filter data for the 'All' group
all_group_data = te_data[te_data['Group'] == 'All']

# Extract relevant data for plotting
plot_data = all_group_data[['Input', 'te_acw_forward', 'te_raw_forward']]

# Compute mean and standard deviation for the TE arrays
plot_data['te_acw_forward_mean'] = plot_data['te_acw_forward'].apply(lambda x: np.mean(eval(x)))
plot_data['te_acw_forward_std'] = plot_data['te_acw_forward'].apply(lambda x: np.std(eval(x)))

plot_data['te_raw_forward_mean'] = plot_data['te_raw_forward'].apply(lambda x: np.mean(eval(x)))
plot_data['te_raw_forward_std'] = plot_data['te_raw_forward'].apply(lambda x: np.std(eval(x)))

# Prepare data for the first plot (Interaction of Input and ACW-0)
te_types = ['te_acw_forward', 'te_raw_forward']
input_colors = {'CS': 'blue', 'WD': 'green'}
input_labels = {'CS': 'Cosine Similarity', 'WD': 'Word Depth'}

means = {input_type: [plot_data[plot_data['Input'] == input_type][f"{te_type}_mean"].mean() for te_type in te_types]
         for input_type in input_colors.keys()}
stds = {input_type: [plot_data[plot_data['Input'] == input_type][f"{te_type}_std"].mean() for te_type in te_types]
        for input_type in input_colors.keys()}

# Prepare data for the second plot (Interaction of Input and ROI)
plot_data_roi = all_group_data[['ROI', 'Input', 'te_acw_forward']]

# Compute mean and standard deviation for the TE arrays (ROI-based)
plot_data_roi['te_acw_forward_mean'] = plot_data_roi['te_acw_forward'].apply(lambda x: np.mean(eval(x)))
plot_data_roi['te_acw_forward_std'] = plot_data_roi['te_acw_forward'].apply(lambda x: np.std(eval(x)))

# Specify the desired order of ROIs
roi_order = ['A1', 'TA2', 'PSL']

# Compute means and standard deviations based on the specified ROI order
means_roi = {input_type: [plot_data_roi[(plot_data_roi['Input'] == input_type) & (plot_data_roi['ROI'] == roi)]['te_acw_forward_mean'].mean()
                         for roi in roi_order]
             for input_type in input_colors.keys()}
stds_roi = {input_type: [plot_data_roi[(plot_data_roi['Input'] == input_type) & (plot_data_roi['ROI'] == roi)]['te_acw_forward_std'].mean()
                        for roi in roi_order]
            for input_type in input_colors.keys()}

# Plotting - creating side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)  # sharey=True to use a common y-axis

# Plot 1: Interaction of Input and ACW-0
x_labels_te = ['Dynamic ACW-0', 'Raw Timeseries']
x_positions_te = range(len(te_types))

for input_type, color in input_colors.items():
    axes[0].errorbar(
        x_positions_te, means[input_type], yerr=stds[input_type],
        label=input_labels[input_type], fmt='s-', color=color, ecolor='black', capsize=3
    )

axes[0].set_title('Interaction of Input and ACW-0', fontweight='bold')
axes[0].set_ylabel('Transfer Entropy', fontweight='bold')
axes[0].set_xticks(x_positions_te)
axes[0].set_xticklabels(x_labels_te)
axes[0].grid(axis='y', alpha=0.5)

# Plot 2: Interaction of Input and ROI (ordered: A1, TA2, PSL)
x_positions_roi = range(len(roi_order))

for input_type, color in input_colors.items():
    axes[1].errorbar(
        x_positions_roi, means_roi[input_type], yerr=stds_roi[input_type],
        label=input_labels[input_type], fmt='s-', color=color, ecolor='black', capsize=3
    )

axes[1].set_title('Interaction of Input and ROI (ACW-0)', fontweight='bold')

axes[1].set_xticks(x_positions_roi)
axes[1].set_xticklabels(roi_order)  # Use the custom order for ROIs
axes[1].grid(axis='y', alpha=0.5)

# Common legend in the top right
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, title='Input', loc='upper right', bbox_to_anchor=(1, 1))

# Tight layout and save the figure
plt.tight_layout()
plt.savefig('interaction.png', dpi=300, bbox_inches='tight')

plt.show()
