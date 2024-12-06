import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import butter, filtfilt
from scipy import signal

# Load the data
#file_path = 'WD_MOVIE2_HOI.csv'  # Adjust the path as necessary
file_path = '6_0-1_7T_MOVIE2_HOI.csv'  # Adjust the path as necessary
data = pd.read_csv(file_path)

# Cap the Word_depth column at 1
time_series = np.clip(data['Cosine_Similarity'], a_min=None, a_max=1)
#time_series = np.clip(data['Word_depth'], a_min=None, a_max=None)
# Remove the first and last 200 samples
time_series = time_series[200:-200].reset_index(drop=True)

# High-pass filter (uncomment if needed)
def apply_highpass(time_series, cutoff=0.002, fs=1.0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(5, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, time_series)

# Optional high-pass filter (apply if needed)
#time_series = apply_highpass(time_series)

def calculate_acw(ts):
    # Detrend the time series
    ts_detrended = signal.detrend(np.array(ts))
    sampling_rate = 1.0
    
    # Compute the autocorrelation function (ACF)
    acw_func = acf(ts_detrended, nlags=len(ts_detrended)-1, qstat=False, alpha=None, fft=True)
    
    # Find the point where the ACF first becomes non-positive
    acw_0 = np.argmax(acw_func <= 0) / sampling_rate
    return acw_0
    # Example usage
acw_value = calculate_acw(time_series)
print(f"ACW0: {acw_value}")


# Sliding window parameters
window_size = 600
step_size = 10
acw0_values = []

# Sliding window calculation
for start in range(0, len(time_series) - window_size + 1, step_size):
    window = time_series[start:start + window_size]
    acw0 = calculate_acw(window)
    acw0_values.append(acw0)
print(len(acw0_values))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Assuming acw0_values, step_size, and rest_intervals are defined

# Apply Gaussian smoothing with sigma = 0.8 to acw0_values
smoothed_acw0_values = gaussian_filter1d(acw0_values, sigma=0.8)

# Plot the dynamic ACW0 time series with adjusted y-values
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, len(acw0_values)), smoothed_acw0_values/step_size, label='Dynamic ACW0', color='b')

# Shade the specified intervals in grey
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]
for start, end in rest_intervals:
    plt.axvspan(start, end, color='grey', alpha=0.5)

# Set the title in bold
plt.title('Dynamic ACW0 of Cosine Similarity', fontweight='bold')

# Set labels
plt.xlabel('Windows (1 time step = 1 second)', fontweight='bold')
plt.ylabel('ACW0', fontweight='bold')

# Add gridlines for the y-axis with dashed lines and alpha transparency
plt.gca().yaxis.grid(True, linestyle='--', alpha=0.5)

# General grid for x-axis (if needed)


# Add legend
#plt.legend()

# Ensure the plot starts at the origin and there's no gap
plt.xlim(0, len(acw0_values))  # Ensure x-axis starts at 0
plt.ylim(bottom=0)  # Ensure y-axis starts at 0

# Save the plot as a PNG file
plt.savefig('CS_ACW0.png', format='png')

# Show the plot
plt.show()
