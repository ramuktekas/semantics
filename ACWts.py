import pandas as pd
import numpy as np
import scipy.signal
from statsmodels.tsa.stattools import acf
'''
COnvert raw timeseries to acw timeseries'''

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
    for i in range(0, len(filtered_ts) - window_size + 1, step_size):
        window = filtered_ts[i:i + window_size]
        acw_0 = calculate_acw(window)
        acw_series.append(acw_0)
    return acw_series

# Function to process a dataset and save to a new CSV
def process_dataset(input_file, output_file):
    subject_ids, time_series_data = load_data(input_file)

    # Prepare an empty DataFrame to store the ACW time series
    acw_df = pd.DataFrame()

    # Process each subject's time series
    for idx in range(len(subject_ids)):
        ts = time_series_data.iloc[idx, :]
        acw_series = get_acw_time_series(ts)

        # Add the subject's ACW time series as a new column, with the subject ID as the header
        acw_df[subject_ids[idx]] = pd.Series(acw_series)

    # Save the resulting DataFrame to a CSV file, with subject IDs as the header
    acw_df.to_csv(output_file, index=False)

# Process the TA2_tpn and PSL_tpn datasets
process_dataset('A1_tpn.csv', 'A1_tantest.csv')
#process_dataset('TA2_tpn.csv', 'TA2_tan.csv')
#process_dataset('PSL_tpn.csv', 'PSL_tan.csv')
