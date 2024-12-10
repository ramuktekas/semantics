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
plt.loglog(cs_positive_freqs, cs_popt[0] * np.power(cs_positive_freqs, cs_popt[1]), color='red', linestyle='--')
plt.ylim(0.01, 1000)  # Fixed y-axis limits
plt.title('Sentence Similarity PSD', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)

# Right subplot: Word Depth
plt.subplot(1, 2, 2)
plt.loglog(wd_frequencies, wd_smoothed_psd, color='green')
plt.loglog(wd_positive_freqs, wd_popt[0] * np.power(wd_positive_freqs, wd_popt[1]), color='red', linestyle='--')
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
