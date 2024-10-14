import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal
from statsmodels.tsa.stattools import acf
import os
from scipy.ndimage import gaussian_filter

# Load the datasets
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    subject_ids = data.iloc[0, :]  # 0th row contains subject IDs
    time_series_data = data.iloc[1:, :].astype(float)  # Remaining rows are time series data
    time_series_data = time_series_data.T  # Transpose so each column is a subject's time series
    return subject_ids, time_series_data

# Functions for processing
def detrend_data(ts):
    return scipy.signal.detrend(ts)

def apply_bandpass(ts):
    low_freq = 0.02
    high_freq = 0.1
    sampling_rate = 1.0
    sos = scipy.signal.butter(N=3, Wn=[low_freq, high_freq], btype='bandpass', fs=sampling_rate, output='sos')
    filtered_ts = scipy.signal.sosfilt(sos, ts)
    return filtered_ts

def calculate_acw(ts):
    sampling_rate = 1.0
    acw_func = acf(ts, nlags=len(ts)-1, qstat=False, alpha=None, fft=True)
    acw_0 = np.argmax(acw_func <= 0) / sampling_rate
    return acw_0

def get_acw_time_series(time_series, window_size=150, step_size=1):
    detrended_ts = detrend_data(time_series)
    filtered_ts = apply_bandpass(detrended_ts)
    acw_series = []
    for i in range(0, len(time_series) - window_size + 1, step_size):
        window = time_series[i:i + window_size]
        acw_0 = calculate_acw(window)
        acw_series.append(acw_0)
    return acw_series

# Define the smoothing function
def smooth_timeseries(ts, sigma):
    return gaussian_filter(ts, sigma=sigma)

# Load the cosine similarity and word depth datasets
cosine_file_path = 'A1_tan.csv'
worddepth_file_path = 'TA2_tan.csv'
PSL_file_path = 'PSL_tan.csv'

cosine_data = pd.read_csv(cosine_file_path)
worddepth_data = pd.read_csv(worddepth_file_path)
PSL_data = pd.read_csv(PSL_file_path)

cosine_similarity_timeseries = cosine_data['A1_tan'].values
word_depth_timeseries = worddepth_data['TA2_tan'].values
PSL_timeseries = PSL_data['PSL_tan'].values


## Calculate dACW time series
#cosine_dacw = get_acw_time_series(cosine_similarity_timeseries)
#word_depth_dacw = get_acw_time_series(word_depth_timeseries)
#
# Apply smoothing (with different parameters for cosine and word depth)
cosine_dacw_smoothed = smooth_timeseries(cosine_similarity_timeseries, sigma=0)
word_depth_dacw_smoothed = smooth_timeseries(word_depth_timeseries, sigma=0)
PSL_tan_smoothed = smooth_timeseries(PSL_timeseries, sigma=0.8)

# Define the movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]

# Directory to save the figures
save_directory = '/Volumes/WD/desktop/Figures8Oct'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# First plot: 2 line plots stacked in 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot Cosine dACW (smoothed)
axs[0].plot(cosine_dacw_smoothed, color='blue', linewidth=1)
axs[0].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[0].set_ylim(5, 6.2)
axs[0].set_title('Dynamic ACW-0 of A1 (Movie run)')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[0].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[0].set_xlim(0, len(cosine_dacw_smoothed))

# Plot Word Depth dACW (smoothed)
axs[1].plot(word_depth_dacw_smoothed, color='green', linewidth=1)

axs[1].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[1].set_ylim(5, 6.2)
axs[1].set_title('Dynamic ACW-0 of TA2 (Movie run)')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[1].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].set_xlim(0, len(word_depth_dacw_smoothed))

# Now, for the third plot, use axs[2] instead of axs[1]
axs[2].plot(PSL_tan_smoothed, color='red', linewidth=1)
axs[2].set_xlabel('Windows (1 time step = 1 second)')
axs[2].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[2].set_ylim(5, 6.2)
axs[2].set_title('Dynamic ACW of PSL (Movie run)')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[2].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid for the third plot
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

axs[2].set_xlim(0, len(PSL_tan_smoothed))

# Save and show the line plots
save_path_line_plots = os.path.join(save_directory, 'dbrain.png')
plt.tight_layout()
plt.savefig(save_path_line_plots, bbox_inches='tight')
plt.show()

print(f"Line plot figure saved at: {save_path_line_plots}")
import matplotlib.pyplot as plt

# Pearson correlation values and labels
correlations = [0.854, 0.467, 0.353]
labels = ['A1-TA2', 'PSL-TA2', 'A1-PSL']

# Define colors
colors = ['orange', 'purple', 'black']

# Create a thin and tall bar plot
fig, ax = plt.subplots(figsize=(2, 8))  # Keep the height tall

# Plot the Pearson correlation values with thicker bars (expanded bar size)
ax.bar(range(len(correlations)), correlations, color=colors, width=1.0)  # Width set to 1.0 for full coverage

# Set y-axis limit and labels on the right side
ax.set_ylim(0, 1)
ax.set_title('Pearson Correlation', pad=10)

# Set ticks on the right side of the plot
ax.yaxis.tick_right()

# Adjust x-axis limits to fully remove gaps
ax.set_xlim(-0.5, 2.5)  # Expanded limits for airtight bars
ax.set_xticks(range(len(correlations)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)

# Remove any padding between the axes and the bars
ax.margins(x=0)

# Turn off any extra space around the plot
ax.autoscale(enable=True, axis='x', tight=True)

# Add horizontal grid and set its transparency
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Save and show the bar plot
save_path = 'braincorr.png'
plt.tight_layout(pad=0)  # Remove any padding around the plot
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Bar plot figure saved at: {save_path}")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal
from statsmodels.tsa.stattools import acf
import os
from scipy.ndimage import gaussian_filter

# Load the datasets
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    subject_ids = data.iloc[0, :]  # 0th row contains subject IDs
    time_series_data = data.iloc[1:, :].astype(float)  # Remaining rows are time series data
    time_series_data = time_series_data.T  # Transpose so each column is a subject's time series
    return subject_ids, time_series_data

# Functions for processing
def detrend_data(ts):
    return scipy.signal.detrend(ts)

def apply_bandpass(ts):
    low_freq = 0.02
    high_freq = 0.1
    sampling_rate = 1.0
    sos = scipy.signal.butter(N=3, Wn=[low_freq, high_freq], btype='bandpass', fs=sampling_rate, output='sos')
    filtered_ts = scipy.signal.sosfilt(sos, ts)
    return filtered_ts

def calculate_acw(ts):
    sampling_rate = 1.0
    acw_func = acf(ts, nlags=len(ts)-1, qstat=False, alpha=None, fft=True)
    acw_0 = np.argmax(acw_func <= 0) / sampling_rate
    return acw_0

def get_acw_time_series(time_series, window_size=150, step_size=1):
    detrended_ts = detrend_data(time_series)
    filtered_ts = apply_bandpass(detrended_ts)
    acw_series = []
    for i in range(0, len(time_series) - window_size + 1, step_size):
        window = time_series[i:i + window_size]
        acw_0 = calculate_acw(window)
        acw_series.append(acw_0)
    return acw_series

# Define the smoothing function
def smooth_timeseries(ts, sigma):
    return gaussian_filter(ts, sigma=sigma)

# Load the cosine similarity and word depth datasets
cosine_file_path = 'A1_rantest.csv'
worddepth_file_path = 'TA2_rantest.csv'
PSL_file_path = 'PSL_rantest.csv'

cosine_data = pd.read_csv(cosine_file_path)
worddepth_data = pd.read_csv(worddepth_file_path)
PSL_data = pd.read_csv(PSL_file_path)

cosine_similarity_timeseries = cosine_data['A1_rpn'].values
word_depth_timeseries = worddepth_data['TA2_rpn'].values
PSL_timeseries = PSL_data['PSL_rpn'].values


## Calculate dACW time series
#cosine_dacw = get_acw_time_series(cosine_similarity_timeseries)
#word_depth_dacw = get_acw_time_series(word_depth_timeseries)
#
# Apply smoothing (with different parameters for cosine and word depth)
cosine_dacw_smoothed = smooth_timeseries(cosine_similarity_timeseries, sigma=0)
word_depth_dacw_smoothed = smooth_timeseries(word_depth_timeseries, sigma=0)
PSL_tan_smoothed = smooth_timeseries(PSL_timeseries, sigma=0.8)

# Define the movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]

# Directory to save the figures
save_directory = '/Volumes/WD/desktop/Figures8Oct'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# First plot: 2 line plots stacked in 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot Cosine dACW (smoothed)
axs[0].plot(cosine_dacw_smoothed, color='blue', linewidth=1)
axs[0].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[0].set_ylim(5.2, 6.5)
axs[0].set_title('Dynamic ACW-0 of A1 (Rest)')

## Add rest intervals as grey boxes
#for rest in rest_intervals:
#    axs[0].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[0].set_xlim(0, len(cosine_dacw_smoothed))

# Plot Word Depth dACW (smoothed)
axs[1].plot(word_depth_dacw_smoothed, color='green', linewidth=1)

axs[1].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[1].set_ylim(5.4, 6.5)
axs[1].set_title('Dynamic ACW-0 of TA2 (Rest)')

## Add rest intervals as grey boxes
#for rest in rest_intervals:
#    axs[1].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].set_xlim(0, len(word_depth_dacw_smoothed))

# Now, for the third plot, use axs[2] instead of axs[1]
axs[2].plot(PSL_tan_smoothed, color='red', linewidth=1)
axs[2].set_xlabel('Windows (1 time step = 1 second)')
axs[2].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[2].set_ylim(5.2, 6.5)
axs[2].set_title('Dynamic ACW of PSL (Rest)')

## Add rest intervals as grey boxes
#for rest in rest_intervals:
#    axs[2].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid for the third plot
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

axs[2].set_xlim(0, len(PSL_tan_smoothed))

# Save and show the line plots
save_path_line_plots = os.path.join(save_directory, 'dbrain.png')
plt.tight_layout()
#plt.savefig(save_path_line_plots, bbox_inches='tight')
plt.show()

print(f"Line plot figure saved at: {save_path_line_plots}")
from scipy.stats import pearsonr

# Compute pairwise Pearson correlations
corr_A1_TA2, _ = pearsonr(cosine_dacw_smoothed, word_depth_dacw_smoothed)
corr_A1_PSL, _ = pearsonr(cosine_dacw_smoothed, PSL_tan_smoothed)
corr_TA2_PSL, _ = pearsonr(word_depth_dacw_smoothed, PSL_tan_smoothed)

# Print the correlation values
print(f"Pearson correlation between A1 and TA2: {corr_A1_TA2:.4f}")
print(f"Pearson correlation between A1 and PSL: {corr_A1_PSL:.4f}")
print(f"Pearson correlation between TA2 and PSL: {corr_TA2_PSL:.4f}")
import matplotlib.pyplot as plt

# Pearson correlation values and labels
correlations = [0.4162, 0.4711, 0.5705]
labels = ['A1-TA2', 'PSL-TA2', 'A1-PSL']

# Define colors
colors = ['orange', 'purple', 'black']

# Create a thin and tall bar plot
fig, ax = plt.subplots(figsize=(2, 8))  # Keep the height tall

# Plot the Pearson correlation values with thicker bars (expanded bar size)
ax.bar(range(len(correlations)), correlations, color=colors, width=1.0)  # Width set to 1.0 for full coverage

# Set y-axis limit and labels on the right side
ax.set_ylim(0, 1)
ax.set_title('Pearson Correlation', pad=10)

# Set ticks on the right side of the plot
ax.yaxis.tick_right()

# Adjust x-axis limits to fully remove gaps
ax.set_xlim(-0.5, 2.5)  # Expanded limits for airtight bars
ax.set_xticks(range(len(correlations)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)

# Remove any padding between the axes and the bars
ax.margins(x=0)

# Turn off any extra space around the plot
ax.autoscale(enable=True, axis='x', tight=True)

# Add horizontal grid and set its transparency
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Save and show the bar plot
save_path = 'braincorr.png'
plt.tight_layout(pad=0)  # Remove any padding around the plot
#plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Bar plot figure saved at: {save_path}")
