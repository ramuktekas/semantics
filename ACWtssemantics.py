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
cosine_file_path = '150_1_7T_MOVIE2_HO1.csv'
worddepth_file_path = 'adjusted_average_wd_intervals.csv'

cosine_data = pd.read_csv(cosine_file_path)
worddepth_data = pd.read_csv(worddepth_file_path)

cosine_similarity_timeseries = cosine_data['Cosine_Similarity'].values
word_depth_timeseries = worddepth_data['average_wd'].values

# Calculate dACW time series
cosine_dacw = get_acw_time_series(cosine_similarity_timeseries)
word_depth_dacw = get_acw_time_series(word_depth_timeseries)

# Apply smoothing (with different parameters for cosine and word depth)
cosine_dacw_smoothed = smooth_timeseries(cosine_dacw, sigma=0.8)
word_depth_dacw_smoothed = smooth_timeseries(word_depth_dacw, sigma=0.8)

# Define the movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]

# Directory to save the figure
save_directory = '/Volumes/WD/desktop/Figures8Oct'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Create subplots: one for Cosine dACW and one for Word Depth dACW
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Cosine dACW (smoothed)
axs[0].plot(cosine_dacw_smoothed, color='blue', linewidth=1)
axs[0].set_ylabel('ACW')
axs[0].set_ylim(0, 50)
axs[0].set_title('Cosine Similarity dACW')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[0].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Set y-axis and origin at the origin
axs[0].spines['left'].set_position(('data', 0))
axs[0].set_xlim(0, len(cosine_dacw_smoothed))

# Plot Word Depth dACW (smoothed)
axs[1].plot(word_depth_dacw_smoothed, color='green', linewidth=1)
axs[1].set_xlabel('Windows (1 time step = 1 second)')
axs[1].set_ylabel('ACW')
axs[1].set_ylim(0, 50)
axs[1].set_title('Word Depth dACW')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[1].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Set y-axis and origin at the origin
axs[1].spines['left'].set_position(('data', 0))
axs[1].set_xlim(0, len(word_depth_dacw_smoothed))

# Save and show the plot
save_path = os.path.join(save_directory, 'dacw_cosine_worddepth_plot.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Figure saved at: {save_path}")
