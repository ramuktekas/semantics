
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, detrend
from statsmodels.tsa.stattools import acf
from scipy.stats import levene
import scipy.signal
from matplotlib.ticker import FuncFormatter

def detrend_data(ts):
    return scipy.signal.detrend(ts)
def calculate_acw(ts):
    sampling_rate = 1.0
#    detrended_ts = detrend_data(ts)
#    filtered_ts = bandpass_filter(ts, 0.05, 0.4999, 1.0)
    acw_func = acf(ts, nlags=len(ts)-1, qstat=False, alpha=None, fft=True)
    acw_0 = np.argmax(acw_func <= 0) / sampling_rate
    return acw_0
def get_acw_time_series(ts, window_size=60, step_size=1):
    detrended_ts = detrend_data(ts)  # Assuming this function is needed
    filtered_ts = bandpass_filter(ts, 0.05, 0.4999, 1.0)  # Apply bandpass filter
    acw_series = []
    
    for i in range(0, len(filtered_ts) - window_size + 1, step_size):  # Use filtered_ts here
        window = filtered_ts[i:i + window_size]  # Use filtered_ts for the window
        acw_0 = calculate_acw(window)  # Assuming this function calculates the ACW for the window
        acw_series.append(acw_0)
    
    return acw_series
# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to process each file and return the mean timeseries
def process_file(filename):
    df = pd.read_csv(filename)
    bold_signals = []
    
    for col in df.columns:  # Ignore the first column (subject ID)
        timeseries = df[col].values[20:-20]  # Remove first and last 20 values
        print(timeseries.shape)
#        detrended = detrend(timeseries)  # Detrend the time series
#        filtered = bandpass_filter(detrended, 0.05, 0.4999, fs=1, order=5)  # Bandpass filter
        acw_series = get_acw_time_series(timeseries)
        bold_signals.append(acw_series)
    print(np.array(bold_signals).shape)
    # Take the mean across all subjects for each timepoint
    mean_bold = np.mean(bold_signals, axis=0)
    return mean_bold

# Load and process files for BOLD signal
mean_A1_rest = process_file("A1_rpn_177subs.csv")
mean_TA2_rest = process_file("TA2_rpn_177subs.csv")
mean_PSL_rest = process_file("PSL_rpn_177subs.csv")

mean_A1_movie = process_file("A1_tpn_182subs.csv")
mean_TA2_movie = process_file("TA2_tpn_182subs.csv")
mean_PSL_movie = process_file("PSL_tpn_182subs.csv")

# Load subject data for Levene's test
rois = ["A1", "TA2", "PSL"]
runs = ["rest1", "movie2"]
rois_colors = {
    "A1": "blue",
    "TA2": "green",
    "PSL": "red"
}

# Initialize result containers
levene_results = {}
data_brain = {run: {roi: [] for roi in rois} for run in runs}

# Extract the subject IDs from the first row (excluding the first column, which is likely the header)
subjects = pd.read_csv("A1_rpn_177subs.csv").columns

for run in runs:
    for roi in rois:
        list_data = []
        # Load and process the data from CSV files
        for subject in subjects:
            filename = f"{roi}_rpn_177subs.csv" if run == "rest1" else f"{roi}_tpn_182subs.csv"
            df = pd.read_csv(filename)
            subject_data = df[subject].values[20:-20]  # Remove first and last 20 values
#            bandpass = bandpass_filter(subject_data, 0.05, 0.4999, fs=1)
            acw_series = get_acw_time_series(subject_data)
            list_data.append(acw_series)
        
        data_brain[run][roi] = np.mean(list_data, axis=0)

# Levene's test
for roi in rois:
    rest_data = data_brain["rest1"][roi]
    movie_data = data_brain["movie2"][roi]
    f_stat, p_value = levene(rest_data, movie_data)
    levene_results[roi] = (f_stat, p_value)
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]
rest_intervals = [(start - 20, end - 20) for start, end in rest_intervals]

# Create the figure with a 3x3 grid and adjusted column widths
fig, axs = plt.subplots(3, 3, figsize=(25, 10), sharey=True, gridspec_kw={'width_ratios': [3.5, 3.5, 1]})

# Plot BOLD time-series for A1, TA2, PSL (Rest and Movie run)
axs[0, 0].plot(mean_A1_rest, color='blue')
axs[0, 0].set_title('Dynamic ACW A1 (Rest)', fontweight='bold')
axs[1, 0].plot(mean_TA2_rest, color='green')
axs[1, 0].set_title('Dynamic ACW TA2 (Rest)', fontweight='bold')
axs[2, 0].plot(mean_PSL_rest[:900], color='red')
axs[2, 0].set_title('BOLD time-series PSL (Rest)', fontweight='bold')

axs[0, 1].plot(mean_A1_movie, color='blue')
axs[0, 1].set_title('Dynamic ACW A1 (Movie run)', fontweight='bold')
axs[1, 1].plot(mean_TA2_movie, color='green')
axs[1, 1].set_title('Dynamic ACW TA2 (Movie run)', fontweight='bold')
axs[2, 1].plot(mean_PSL_movie[:900], color='red')
axs[2, 1].set_title('Dynamic ACW PSL (Movie run)', fontweight='bold')

axes_list = [
    axs[0, 0], axs[0, 1], axs[0, 2],
    axs[1, 0], axs[1, 1], axs[1, 2],
    axs[2, 0], axs[2, 1], axs[2, 2]
]
for ax in [axs[0, 2], axs[1, 2], axs[2, 2]]:
    ax.set_xticklabels(["Rest", "Movie run"], fontweight="bold", fontsize=10)

# Ensure x-axis starts from 0 in columns 0 and 1
for ax in [axs[0, 1], axs[1, 1], axs[2, 1], axs[0, 0], axs[1, 0], axs[2, 0]]:
    ax.set_xlim(0, len(mean_A1_rest) * 1.05)  # Extend x-axis limit by 5%
    ax.set_xlabel('Windows (1 step = 1 second)', fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(0, len(mean_A1_movie), 100))  # Set tick intervals for x-axis

#    ax.set_ylabel('BOLD', fontsize=12, fontweight='bold')

for ax in axes_list:
    ax.set_xlim(0, len(mean_A1_movie))
#    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))  # Format y-ticks as integers
#    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')  # Ensure x-axis label
    ax.set_ylabel('Autocorrelation Window (ACW) (seconds)', fontsize=7, fontweight='bold')  # Ensure y-axis label
#    ax.set_ylim(2.4, 3.8)  # Set the y-axis limits for consistency

#    ax.set_yticks(np.arange(2.4, 3.9, 0.2))  # Set the tick intervals for y-axis
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
for ax in [axs[0, 1], axs[1, 1], axs[2, 1]]:
    for rest in rest_intervals:
        ax.axvspan(rest[0], rest[1], color='grey', alpha=0.5)
# Plot Levene's test results for A1, TA2, PSL
for i, (roi, ax) in enumerate(zip(rois, axs[:, 2])):
    rest_data = data_brain["rest1"][roi]
    movie_data = data_brain["movie2"][roi]
    
    # Boxplot and stripplot
    data = [rest_data, movie_data]
    boxplots = ax.boxplot(data, positions=[0, 1], patch_artist=True, widths=0.6, showfliers=False)
    for box, color in zip(boxplots["boxes"], [rois_colors[roi], rois_colors[roi]]):
        box.set_facecolor(color)
    sns.stripplot(data=data, jitter=0.3, palette=[rois_colors[roi], rois_colors[roi]], size=2, ax=ax)
    
    f_stat, p_value = levene_results[roi]
    significance = ""
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    
    ax.text(0.5, 3.4, f"F={f_stat:.2f}\n{significance}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"Levene's test {roi}", fontweight='bold', fontsize=10)
#    ax.set_ylim(2.4, 3.8)  # Set the y-axis limits for consistency

#    ax.set_yticks(np.arange(2.4, 3.9, 0.2))  # Set the tick intervals for y-axis

    ax.set_xlim(-0.5, 1.5)
#    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
#    ax.set_ylabel('BOLD', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Adjust spacing
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Add horizontal grids to all plots
for ax in axs.flat:
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Save or show the figure
plt.tight_layout()
fig.savefig('BOLD_Levene_ACW.png', bbox_inches='tight')
plt.show()
