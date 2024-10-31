import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to process each file and return the mean timeseries
def process_file(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :-1]  # Ignore last column
    bold_signals = []

    for col in df.columns[1:]:  # Ignore the first row (subject ID)
        timeseries = df[col].values
        detrended = detrend(timeseries)
        filtered = bandpass_filter(detrended, 0.02, 0.1, fs=1, order=3)
        bold_signals.append(filtered)
    
    # Take the mean across all subjects for each timepoint
    mean_bold = np.mean(bold_signals, axis=0)
    return mean_bold

# Load and process files
mean_A1 = process_file("A1_tpn.csv")
mean_TA2 = process_file("TA2_tpn.csv")
mean_PSL = process_file("PSL_tpn.csv")

# Movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]

# Generate a time axis (assuming 900 time points in each timeseries)
time = np.arange(900)  # X axis goes from 0 to 900 seconds

# Plot setup with 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

# Plotting A1 BOLD signal
axs[0].plot(time, mean_A1[:900], color='blue')
axs[0].set_title('BOLD Signal A1 (Movie run)',fontweight='bold')
#axs[0].set_ylabel('BOLD Signal')

# Plotting TA2 BOLD signal
axs[1].plot(time, mean_TA2[:900], color='green')
axs[1].set_title('BOLD Signal TA2 (Movie run)',fontweight='bold')
#axs[1].set_ylabel('BOLD Signal')

# Plotting PSL BOLD signal
axs[2].plot(time, mean_PSL[:900], color='red')
axs[2].set_title('BOLD Signal PSL (Movie run)',fontweight='bold')
#axs[2].set_ylabel('BOLD Signal')
axs[2].set_xlabel('Time (seconds)',fontweight='bold')
fig.supylabel('BOLD signal', fontsize=12, fontweight='bold')
# Add grey boxes for rest intervals on all plots
for ax in axs:
    for rest in rest_intervals:
        ax.axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Set Y-axis limits to be the same across all plots
max_y = max(mean_A1.max(), mean_TA2.max(), mean_PSL.max())
min_y = min(mean_A1.min(), mean_TA2.min(), mean_PSL.min())
for ax in axs:
    ax.set_ylim([min_y, max_y])
    
# Add horizontal grid to all plots
for ax in axs:
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Limit X-axis to 0-900 seconds
plt.xlim(0, 900)

# Save the first plot as 'Braintask.png'
plt.tight_layout()
braintask_save_path = 'movie.png'
#plt.savefig(braintask_save_path, bbox_inches='tight')

# Display plot
plt.show()


from scipy.stats import pearsonr

# Compute pairwise Pearson correlations
corr_A1_TA2, _ = pearsonr(mean_A1[:900], mean_TA2[:900])
corr_A1_PSL, _ = pearsonr(mean_A1[:900], mean_PSL[:900])
corr_TA2_PSL, _ = pearsonr(mean_TA2[:900], mean_PSL[:900])

# Print the correlation values
print(f"Pearson correlation between A1 and TA2: {corr_A1_TA2:.4f}")
print(f"Pearson correlation between A1 and PSL: {corr_A1_PSL:.4f}")
print(f"Pearson correlation between TA2 and PSL: {corr_TA2_PSL:.4f}")

# Second plot: Thin and tall bar plot
fig, ax = plt.subplots(figsize=(2, 8))  # Keep the height tall

# Pearson correlation values between the last columns of the datasets
correlations = [0.8953, 0.3830, 0.3473]
labels = ['A1-TA2', 'PSL-TA2', 'A1-PSL']

# Define colors
colors = ['orange', 'purple', 'black']

# Create a thinner and tall bar plot
fig, ax = plt.subplots(figsize=(2, 8))  # Reduced width for thinner plot

# Plot the Pearson correlation values with full-width bars for no spacing
ax.bar(range(len(correlations)), correlations, color=colors, width=1.0)  # Full-width bars

# Set y-axis limit and labels on the right side
ax.set_ylim(0, 1)

# Adjust title position by setting 'loc' to 'left' and fine-tuning pad
ax.set_title('Pearson Correlation', pad=20, fontweight='bold', loc='left')  # Move title slightly to the right

# Set ticks on the right side of the plot
ax.yaxis.tick_right()

# Adjust x-axis limits tightly around the bars to remove gaps
ax.set_xlim(-0.5, len(correlations) - 0.5)  # Tighten x-axis limits based on number of bars
ax.set_xticks(range(len(correlations)))
ax.set_xticklabels(labels, rotation=90, fontsize=8, fontweight='bold')

# Remove any padding between the axes and the bars
ax.margins(x=0)

# Use tight layout to ensure no cutoff
plt.tight_layout()

# Add horizontal grid and set its transparency
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Save and show the bar plot
save_path = 'cor_movie.png'
#plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Bar plot figure saved at: {save_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to process each file and return the mean timeseries
def process_file(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :-1]  # Ignore last column
    bold_signals = []

    for col in df.columns[1:]:  # Ignore the first row (subject ID)
        timeseries = df[col].values
        detrended = detrend(timeseries)
        filtered = bandpass_filter(detrended, 0.02, 0.1, fs=1, order=3)
        bold_signals.append(filtered)
    
    # Take the mean across all subjects for each timepoint
    mean_bold = np.mean(bold_signals, axis=0)
    return mean_bold

# Load and process files
mean_A1 = process_file("A1_rpn.csv")
mean_TA2 = process_file("TA2_rpn.csv")
mean_PSL = process_file("PSL_rpn.csv")

# Movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]

# Generate a time axis (assuming 900 time points in each timeseries)
time = np.arange(900)  # X axis goes from 0 to 900 seconds

# Plot setup with 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

# Plotting A1 BOLD signal
axs[0].plot(time, mean_A1[:900], color='blue')
axs[0].set_title('BOLD Signal A1 (Rest)',fontweight='bold')
#axs[0].set_ylabel('BOLD Signal')

# Plotting TA2 BOLD signal
axs[1].plot(time, mean_TA2[:900], color='green')
axs[1].set_title('BOLD Signal TA2 (Rest)',fontweight='bold')
#axs[1].set_ylabel('BOLD Signal')

# Plotting PSL BOLD signal
axs[2].plot(time, mean_PSL[:900], color='red')
axs[2].set_title('BOLD Signal PSL (Rest)',fontweight='bold')
#axs[2].set_ylabel('BOLD Signal')
axs[2].set_xlabel('Time (seconds)')
fig.supylabel('BOLD signal', fontsize=12, fontweight='bold')

## Add grey boxes for rest intervals on all plots
#for ax in axs:
#    for rest in rest_intervals:
#        ax.axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Set Y-axis limits to be the same across all plots
# Calculate the range of y-axis based on the data
max_y = max(mean_A1.max(), mean_TA2.max(), mean_PSL.max())
min_y = min(mean_A1.min(), mean_TA2.min(), mean_PSL.min())

# Set y-limits and customize y-ticks without -100 and 100
for ax in axs:
    ax.set_ylim([-100, 100])
    # Get current ticks and remove -100 and 100
    ticks = [tick for tick in ax.get_yticks() if tick != -100 and tick != 100]
    ax.set_yticks(ticks)  # Update the y-ticks

    
# Add horizontal grid to all plots
for ax in axs:
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Limit X-axis to 0-900 seconds
plt.xlim(0, 900)

# Save the first plot as 'Braintask.png'
plt.tight_layout()
braintask_save_path = 'rest.png'
#plt.savefig(braintask_save_path, bbox_inches='tight')

# Display plot
plt.show()


from scipy.stats import pearsonr

# Compute pairwise Pearson correlations
corr_A1_TA2, _ = pearsonr(mean_A1[:900], mean_TA2[:900])
corr_A1_PSL, _ = pearsonr(mean_A1[:900], mean_PSL[:900])
corr_TA2_PSL, _ = pearsonr(mean_TA2[:900], mean_PSL[:900])

# Print the correlation values
print(f"Pearson correlation between A1 and TA2: {corr_A1_TA2:.4f}")
print(f"Pearson correlation between A1 and PSL: {corr_A1_PSL:.4f}")
print(f"Pearson correlation between TA2 and PSL: {corr_TA2_PSL:.4f}")

# Second plot: Thin and tall bar plot
fig, ax = plt.subplots(figsize=(2, 8))  # Keep the height tall

# Pearson correlation values between the last columns of the datasets
correlations = [0.7571, 0.3882, 0.6094]
labels = ['A1-TA2', 'PSL-TA2', 'A1-PSL']

# Define colors
colors = ['orange', 'purple', 'black']

# Create a thinner and tall bar plot
fig, ax = plt.subplots(figsize=(2, 8))  # Reduced width for thinner plot

# Plot the Pearson correlation values with full-width bars for no spacing
ax.bar(range(len(correlations)), correlations, color=colors, width=1.0)  # Full-width bars

# Set y-axis limit and labels on the right side
ax.set_ylim(0, 1)

# Adjust title position by setting 'loc' to 'left' and fine-tuning pad
ax.set_title('Pearson Correlation', pad=20, fontweight='bold', loc='left')  # Move title slightly to the right

# Set ticks on the right side of the plot
ax.yaxis.tick_right()

# Adjust x-axis limits tightly around the bars to remove gaps
ax.set_xlim(-0.5, len(correlations) - 0.5)  # Tighten x-axis limits based on number of bars
ax.set_xticks(range(len(correlations)))
ax.set_xticklabels(labels, rotation=90, fontsize=8, fontweight='bold')

# Remove any padding between the axes and the bars
ax.margins(x=0)

# Use tight layout to ensure no cutoff
plt.tight_layout()

# Add horizontal grid and set its transparency
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Save and show the bar plot
save_path = 'cor_rest.png'
#plt.savefig(save_path, bbox_inches='tight')
plt.show()
