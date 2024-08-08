import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Define the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Function to calculate ACW time series
def calculate_acw_ts(data, window_size, step_size, fs=1):
    n_windows = (len(data) - window_size) // step_size + 1
    acw_ts = []
    for i in range(n_windows):
        window_data = data[i*step_size:i*step_size + window_size]
        acf_values = acf(window_data, nlags=len(window_data) - 1, fft=True)
        acw = np.where(acf_values < 0)[0]
        acw_ts.append(acw[0] if len(acw) > 0 else len(acf_values))
    return np.array(acw_ts)

# Parameters
lowcut = 0.02
highcut = 0.1
fs = 1  # Sampling frequency
window_size = 150  # 150 seconds
step_size = 1  # 1 second

# Preprocess and calculate ACW time series for A1, TA2, and PSL data
def process_subject_time_series(data, lowcut, highcut, fs, window_size, step_size):
    subjects_acw_ts = []
    for subject_id in data.columns[:-1]:  # Exclude the last column which is the average
        ts = data[subject_id].values
        ts_detrended = detrend(ts)
        ts_filtered = bandpass_filter(ts_detrended, lowcut, highcut, fs)
        acw_ts = calculate_acw_ts(ts_filtered, window_size, step_size, fs)
        subjects_acw_ts.append(acw_ts)
    return np.mean(subjects_acw_ts, axis=0)

# Load data
a1_tpn = pd.read_csv('A1_tpn.csv')
ta2_tpn = pd.read_csv('TA2_tpn.csv')
psl_tpn = pd.read_csv('PSL_tpn.csv')
a1_rpn = pd.read_csv('A1_rpn.csv')
ta2_rpn = pd.read_csv('TA2_rpn.csv')
psl_rpn = pd.read_csv('PSL_rpn.csv')

# Calculate average ACW time series for task condition
acw_ts_a1_task_corrected = process_subject_time_series(a1_tpn, lowcut, highcut, fs, window_size, step_size)
acw_ts_ta2_task_corrected = process_subject_time_series(ta2_tpn, lowcut, highcut, fs, window_size, step_size)
acw_ts_psl_task_corrected = process_subject_time_series(psl_tpn, lowcut, highcut, fs, window_size, step_size)

# Calculate average ACW time series for rest condition
acw_ts_a1_rest_corrected = process_subject_time_series(a1_rpn, lowcut, highcut, fs, window_size, step_size)
acw_ts_ta2_rest_corrected = process_subject_time_series(ta2_rpn, lowcut, highcut, fs, window_size, step_size)
acw_ts_psl_rest_corrected = process_subject_time_series(psl_rpn, lowcut, highcut, fs, window_size, step_size)

# Plotting the corrected ACW time series for task condition
plt.figure(figsize=(18, 12))

# Plot A1 Task
plt.subplot(3, 1, 1)
plt.plot(acw_ts_a1_task_corrected, color='blue')
plt.grid(False)
plt.title('ACW Time Series for A1 Task')

# Plot TA2 Task
plt.subplot(3, 1, 2)
plt.plot(acw_ts_ta2_task_corrected, color='green')
plt.grid(False)
plt.title('ACW Time Series for TA2 Task')

# Plot PSL Task
plt.subplot(3, 1, 3)
plt.plot(acw_ts_psl_task_corrected, color='red')
plt.grid(False)
plt.title('ACW Time Series for PSL Task')

plt.tight_layout()
plt.show()

# Plotting the corrected ACW time series for rest condition
plt.figure(figsize=(18, 12))

# Plot A1 Rest
plt.subplot(3, 1, 1)
plt.plot(acw_ts_a1_rest_corrected, color='blue')
plt.grid(False)
plt.title('ACW Time Series for A1 Rest')

# Plot TA2 Rest
plt.subplot(3, 1, 2)
plt.plot(acw_ts_ta2_rest_corrected, color='green')
plt.grid(False)
plt.title('ACW Time Series for TA2 Rest')

# Plot PSL Rest
plt.subplot(3, 1, 3)
plt.plot(acw_ts_psl_rest_corrected, color='red')
plt.grid(False)
plt.title('ACW Time Series for PSL Rest')

plt.tight_layout()
plt.show()
