import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('Transfer_entropy.csv')

# Filter the data to keep only A1
selected_data_group_all = data[(data['Group'] == 'All')]
a1_data_cs = selected_data_group_all[(selected_data_group_all['ROI'] == 'A1') & (selected_data_group_all['Input'].str.lower() == 'cs')]
a1_data_wd = selected_data_group_all[(selected_data_group_all['ROI'] == 'A1') & (selected_data_group_all['Input'].str.lower() == 'wd')]

# Lighter shade for reverse
light_blue = '#add8e6'

# Prepare the plot with shared Y-axis for both CS and WD, showing forward and reverse with different shades
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Plot for CS (A1)
for index, row in a1_data_cs.iterrows():
    te_acw_forward_values = eval(row['te_acw_forward'])
    te_raw_forward_values = eval(row['te_raw_forward'])
    te_acw_reverse_values = eval(row['te_acw_reverse'])
    te_raw_reverse_values = eval(row['te_raw_reverse'])

    ax1.plot(range(1, 11), te_acw_forward_values, marker='o', color='blue', label='ACW forward')
    ax1.plot(range(1, 11), te_raw_forward_values, marker='^', linestyle='--', color='blue', label='RAW forward')
    ax1.plot(range(1, 11), te_acw_reverse_values, marker='o', color=light_blue, label='ACW reverse')
    ax1.plot(range(1, 11), te_raw_reverse_values, marker='^', linestyle='--', color=light_blue, label='RAW reverse')

ax1.set_title('Transfer Entropy (Cosine Similarity) - A1')
ax1.set_xlabel('Lag (seconds)')
ax1.set_ylabel('Transfer Entropy')
ax1.set_xticks(range(1, 11))
ax1.legend()
ax1.grid(False)

# Plot for WD (A1)
for index, row in a1_data_wd.iterrows():
    te_acw_forward_values = eval(row['te_acw_forward'])
    te_raw_forward_values = eval(row['te_raw_forward'])
    te_acw_reverse_values = eval(row['te_acw_reverse'])
    te_raw_reverse_values = eval(row['te_raw_reverse'])

    ax2.plot(range(1, 11), te_acw_forward_values, marker='o', color='blue', label='ACW forward')
    ax2.plot(range(1, 11), te_raw_forward_values, marker='^', linestyle='--', color='blue', label='RAW forward')
    ax2.plot(range(1, 11), te_acw_reverse_values, marker='o', color=light_blue, label='ACW reverse')
    ax2.plot(range(1, 11), te_raw_reverse_values, marker='^', linestyle='--', color=light_blue, label='RAW reverse')

ax2.set_title('Transfer Entropy (Word Depths) - A1')
ax2.set_xlabel('Lag (seconds)')
ax2.set_xticks(range(1, 11))
ax2.legend()
ax2.grid(False)

# Adjust layout
plt.tight_layout()
plt.show()
