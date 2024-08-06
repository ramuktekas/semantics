import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

# Load the cosine similarity timeseries
cosine_file_path = '150_1_7T_MOVIE2_HO1.csv'
cosine_data = pd.read_csv(cosine_file_path)
cosine_similarity_timeseries = cosine_data['Cosine_Similarity'].values

# Compute the autocorrelation function
acf_values = acf(cosine_similarity_timeseries, nlags=len(cosine_similarity_timeseries) - 1, fft=True)

# Calculate the Power Spectral Density using Welch's method
frequencies, psd = welch(cosine_similarity_timeseries, scaling='density')

# Fit a power law to the PSD
def power_law(x, a, b):
    return a * np.power(x, b)

# Use only the positive, non-zero frequencies and corresponding PSD values for fitting
positive_freqs = frequencies[frequencies > 0]
positive_psd = psd[frequencies > 0]

# Fit the power law
popt, pcov = curve_fit(power_law, positive_freqs, positive_psd)

# Extract the power law parameters
a, b = popt

# Monte Carlo simulation for statistical significance
n_iterations = 1000
monte_carlo_exponents = []

for _ in range(n_iterations):
    # Shuffle the time series
    shuffled_series = np.random.permutation(cosine_similarity_timeseries)
    
    # Calculate the PSD of the shuffled series
    _, shuffled_psd = welch(shuffled_series, scaling='density')
    
    # Fit the power law to the shuffled PSD
    shuffled_positive_psd = shuffled_psd[frequencies > 0]
    popt_shuffled, _ = curve_fit(power_law, positive_freqs, shuffled_positive_psd)
    
    # Store the exponent
    monte_carlo_exponents.append(popt_shuffled[1])

# Calculate p-value
p_value_mc = np.sum(np.array(monte_carlo_exponents) <= b) / n_iterations

# Log-transform the positive frequencies and corresponding PSD values
log_freqs = np.log10(positive_freqs)
log_psd = np.log10(positive_psd)

# Perform linear regression on the log-transformed data
slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_psd)

# Calculate the fitted values
fitted_log_psd = slope * log_freqs + intercept

# Plotting all three figures separately

# Plot 1: Cosine Similarity Time Series
plt.figure(figsize=(10, 4))
plt.plot(cosine_similarity_timeseries, color='blue')
plt.grid(False)
plt.title('Cosine Similarity Time Series')
plt.show()

# Plot 2: Autocorrelation Function
plt.figure(figsize=(10, 4))
plt.plot(acf_values, color='blue')
plt.fill_between(range(len(acf_values)), acf_values, color='lightblue')
plt.grid(False)
plt.title('Autocorrelation Function')
plt.show()

# Plot 3: Power Spectral Density
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, psd, label='PSD', color='blue')
plt.loglog(positive_freqs, 10**fitted_log_psd, label=f'Linear Fit: $y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r_value**2:.4f}$\n$p$-value (fit): {p_value:.4f}$\n$p$-value (MC): {p_value_mc:.4f}$', color='red')
plt.legend()
plt.grid(False)
plt.title('Power Spectral Density')
plt.show()

# Plotting all the figures combined
plt.figure(figsize=(18, 12))

# Combined Plot 1: Cosine Similarity Time Series
plt.subplot(3, 1, 1)
plt.plot(cosine_similarity_timeseries, color='blue')
plt.grid(False)
plt.title('Cosine Similarity Time Series')

# Combined Plot 2: Autocorrelation Function
plt.subplot(3, 1, 2)
plt.plot(acf_values, color='blue')
plt.fill_between(range(len(acf_values)), acf_values, color='lightblue')
plt.grid(False)
plt.title('Autocorrelation Function')

# Combined Plot 3: Power Spectral Density
plt.subplot(3, 1, 3)
plt.loglog(frequencies, psd, label='PSD', color='blue')
plt.loglog(positive_freqs, 10**fitted_log_psd, label=f'Linear Fit: $y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r_value**2:.4f}$\n$p$-value (fit): {p_value:.4f}$\n$p$-value (MC): {p_value_mc:.4f}$', color='red')
plt.legend()
plt.grid(False)
plt.title('Power Spectral Density')

plt.tight_layout()
plt.show()

# Print statistical results
print(f"Power Law Fit Equation: y = {slope:.2f}x + {intercept:.2f}")
print(f"R^2 Value: {r_value**2:.4f}")
print(f"P-value (fit): {p_value:.4f}")
print(f"P-value (Monte Carlo): {p_value_mc:.4f}")
