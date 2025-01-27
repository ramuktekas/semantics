'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.ndimage import gaussian_filter1d

# Define the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Parameters
lowcut = 0.05
highcut = 0.4999
fs = 1  # Sampling frequency

# Preprocess and compute PSD for each subject
def compute_subject_psds(data, lowcut, highcut, fs):
    frequencies_list = []
    psd_list = []
    for subject_id in data.columns:
        ts = data[subject_id].values[20:-20]  # Remove 20 values from the front and end
        ts_detrended = detrend(ts)
        ts_filtered = bandpass_filter(ts_detrended, lowcut, highcut, fs)
        frequencies, psd = welch(ts_filtered, fs=fs, scaling='density')
        frequencies_list.append(frequencies)
        psd_list.append(psd)
    
    avg_psd = np.mean(psd_list, axis=0)  # Average PSDs across subjects
    return frequencies_list[0], avg_psd

# Load data
a1_tpn = pd.read_csv('A1_tpn_182subs.csv', index_col=0)
ta2_tpn = pd.read_csv('TA2_tpn_182subs.csv', index_col=0)
psl_tpn = pd.read_csv('PSL_tpn_182subs.csv', index_col=0)

a1_rpn = pd.read_csv('A1_rpn_177subs.csv', index_col=0)
ta2_rpn = pd.read_csv('TA2_rpn_177subs.csv', index_col=0)
psl_rpn = pd.read_csv('PSL_rpn_177subs.csv', index_col=0)

# Find common subjects between Task and Rest datasets
common_subjects = list(set(a1_tpn.columns) & set(a1_rpn.columns))

# Filter datasets to include only common subjects
a1_tpn = a1_tpn[common_subjects]
ta2_tpn = ta2_tpn[common_subjects]
psl_tpn = psl_tpn[common_subjects]

a1_rpn = a1_rpn[common_subjects]
ta2_rpn = ta2_rpn[common_subjects]
psl_rpn = psl_rpn[common_subjects]

# Compute PSDs for Task and Rest
a1_tpn_freqs, a1_tpn_psd = compute_subject_psds(a1_tpn, lowcut, highcut, fs)
ta2_tpn_freqs, ta2_tpn_psd = compute_subject_psds(ta2_tpn, lowcut, highcut, fs)
psl_tpn_freqs, psl_tpn_psd = compute_subject_psds(psl_tpn, lowcut, highcut, fs)

a1_rpn_freqs, a1_rpn_psd = compute_subject_psds(a1_rpn, lowcut, highcut, fs)
ta2_rpn_freqs, ta2_rpn_psd = compute_subject_psds(ta2_rpn, lowcut, highcut, fs)
psl_rpn_freqs, psl_rpn_psd = compute_subject_psds(psl_rpn, lowcut, highcut, fs)

# Plot Task vs. Rest PSD
plt.figure(figsize=(14, 6))

# Task PSD plot
plt.subplot(1, 2, 2)
plt.plot(a1_tpn_freqs, a1_tpn_psd, label='A1', color='blue')
plt.plot(ta2_tpn_freqs, ta2_tpn_psd, label='TA2', color='green')
plt.plot(psl_tpn_freqs, psl_tpn_psd, label='PSL', color='red')
plt.title('PSD of BOLD time-series - Movie Run', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
plt.ylim(0, 42000)
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.xlim(0.05, 0.5)  # Set x-axis limits
plt.legend()

# Rest PSD plot
plt.subplot(1, 2, 1)
plt.plot(a1_rpn_freqs, a1_rpn_psd, label='A1', color='blue')
plt.plot(ta2_rpn_freqs, ta2_rpn_psd, label='TA2', color='green')
plt.plot(psl_rpn_freqs, psl_rpn_psd, label='PSL', color='red')
plt.title('PSD of BOLD time-series - Rest', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.xlim(0.05, 0.5)  # Set x-axis limits
plt.ylim(0, 42000)
plt.legend()

# Save the plot as a PNG file
plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to make room for the y-axis label
plt.savefig('PSD_BOLD.png')

# Show the plot
plt.show()
'''

import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Define a function to process data and compute smoothed PSD and power-law fit
def process_data(file_path, column_name):
    data = pd.read_csv(file_path)
    timeseries = data[column_name].values

    # Remove the first and last 200 samples
    timeseries = timeseries[200:-200]

    # Calculate the Power Spectral Density using Welch's method
    frequencies, psd = welch(timeseries, scaling='density')

    # Apply Gaussian smoothing with kernel size 1.5
    smoothed_psd = gaussian_filter1d(psd, sigma=1.5)

    # Fit a power law to the PSD
    def power_law(x, a, b):
        return a * np.power(x, b)

    # Use only the positive, non-zero frequencies and corresponding PSD values for fitting
    positive_freqs = frequencies[frequencies > 0]
    positive_psd = smoothed_psd[frequencies > 0]

    # Fit the power law
    popt, _ = curve_fit(power_law, positive_freqs, positive_psd)

    return frequencies, smoothed_psd, positive_freqs, popt

# Process Word Depth data
wd_file_path = 'WD_MOVIE2_HOI.csv'
wd_column_name = 'Word_depth'
wd_frequencies, wd_smoothed_psd, wd_positive_freqs, wd_popt = process_data(wd_file_path, wd_column_name)

# Process Cosine Similarity data
cs_file_path = '1_0-1_7T_MOVIE2_HOI.csv'
cs_column_name = 'Cosine_Similarity'
cs_frequencies, cs_smoothed_psd, cs_positive_freqs, cs_popt = process_data(cs_file_path, cs_column_name)

# Create the plot
plt.figure(figsize=(14, 6))

# Left subplot: Cosine Similarity
plt.subplot(1, 2, 1)
plt.loglog(cs_frequencies, cs_smoothed_psd, color='blue')
plt.ylim(0.01, 1000)  # Fixed y-axis limits
plt.title('Sentence Similarity PSD', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)

# Right subplot: Word Depth
plt.subplot(1, 2, 2)
plt.loglog(wd_frequencies, wd_smoothed_psd, color='green')
plt.ylim(0.01, 1000)  # Fixed y-axis limits
plt.title('Word Depth PSD', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)

# Common y-axis label
fig = plt.gcf()
fig.text(0.04, 0.5, 'Power Spectral Density', fontsize=14, fontweight='bold', va='center', rotation='vertical')

# Save the plot as a PNG file
plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to make room for the y-axis label
plt.savefig('PSD_semantics.png')

# Show the plot
plt.show()
