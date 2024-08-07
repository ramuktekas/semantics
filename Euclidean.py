import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean

# Load your data
file_path = 'adjusted_average_wd_intervals.csv'
data = pd.read_csv(file_path)
cosine_similarity_timeseries = data['average_wd'].values

# Ensure the original time series is a 1-D array and remove NaN or Inf values
cosine_similarity_timeseries = np.array(cosine_similarity_timeseries).flatten()
cosine_similarity_timeseries = cosine_similarity_timeseries[~np.isnan(cosine_similarity_timeseries)]
cosine_similarity_timeseries = cosine_similarity_timeseries[~np.isinf(cosine_similarity_timeseries)]
print(f"Length of the time series: {len(cosine_similarity_timeseries)}")

# Check if the cleaned time series is empty
if len(cosine_similarity_timeseries) == 0:
    raise ValueError("The cleaned time series is empty after removing NaN and Inf values.")

# Define the Markov Block Bootstrap function
def markov_block_bootstrap(time_series, sampling_rate=1.0):
    acw_func = acf(time_series, nlags=len(time_series) - 1, fft=True)
    block_length = np.argmax(acw_func <= 0) / sampling_rate
    block_length = max(int(block_length), 1)
    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]
    shuffled_blocks = shuffle(blocks, random_state=None)
    shuffled_time_series = np.concatenate(shuffled_blocks)
    return shuffled_time_series

# Define the Philipp Shuffle function
def philipp_shuffle(time_series, sampling_rate=1.0):
    acw_func = acf(time_series, nlags=len(time_series) - 1, fft=True)
    block_length = np.argmax(acw_func <= 0) / sampling_rate
    block_length = max(int(block_length), 1)
    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]
    shuffled_blocks = [np.random.permutation(block) for block in blocks]
    shuffled_time_series = np.concatenate(shuffled_blocks)
    return shuffled_time_series

# Define the Random Shuffle function
def random_shuffle(time_series):
    return np.random.permutation(time_series)

# Perform shuffling and calculate Euclidean distance
n_iterations = 1000
euclidean_distances = {
    'markov': [],
    'philipp': [],
    'random': []
}
shuffled_series = {
    'markov': [],
    'philipp': [],
    'random': []
}

for i in range(n_iterations):
    markov_shuffled = markov_block_bootstrap(cosine_similarity_timeseries)
    philipp_shuffled = philipp_shuffle(cosine_similarity_timeseries)
    random_shuffled = random_shuffle(cosine_similarity_timeseries)
    
    markov_shuffled = np.array(markov_shuffled).flatten()
    philipp_shuffled = np.array(philipp_shuffled).flatten()
    random_shuffled = np.array(random_shuffled).flatten()
    
    # Check for NaN or Inf values in shuffled data
    if np.any(np.isnan(markov_shuffled)) or np.any(np.isinf(markov_shuffled)):
        raise ValueError(f"Markov shuffled time series contains NaN or Inf values at iteration {i+1}.")
    if np.any(np.isnan(philipp_shuffled)) or np.any(np.isinf(philipp_shuffled)):
        raise ValueError(f"Philipp shuffled time series contains NaN or Inf values at iteration {i+1}.")
    if np.any(np.isnan(random_shuffled)) or np.any(np.isinf(random_shuffled)):
        raise ValueError(f"Random shuffled time series contains NaN or Inf values at iteration {i+1}.")
    
    # Normalize the data
    markov_shuffled = (markov_shuffled - np.mean(markov_shuffled)) / np.std(markov_shuffled)
    philipp_shuffled = (philipp_shuffled - np.mean(philipp_shuffled)) / np.std(philipp_shuffled)
    random_shuffled = (random_shuffled - np.mean(random_shuffled)) / np.std(random_shuffled)
    normalized_cosine_similarity = (cosine_similarity_timeseries - np.mean(cosine_similarity_timeseries)) / np.std(cosine_similarity_timeseries)
    
    # Ensure inputs are correctly formatted and are 1-D arrays
    if normalized_cosine_similarity.ndim != 1 or markov_shuffled.ndim != 1:
        raise ValueError(f"Shape mismatch: {normalized_cosine_similarity.shape} vs {markov_shuffled.shape}")

    print(f"Computing Euclidean distance for iteration {i+1}")
    markov_distance = euclidean(normalized_cosine_similarity, markov_shuffled)
    philipp_distance = euclidean(normalized_cosine_similarity, philipp_shuffled)
    random_distance = euclidean(normalized_cosine_similarity, random_shuffled)
    
    euclidean_distances['markov'].append(markov_distance)
    euclidean_distances['philipp'].append(philipp_distance)
    euclidean_distances['random'].append(random_distance)
    
    shuffled_series['markov'].append(markov_shuffled)
    shuffled_series['philipp'].append(philipp_shuffled)
    shuffled_series['random'].append(random_shuffled)

# Save Euclidean distances to a CSV file
euclidean_df = pd.DataFrame(euclidean_distances)
euclidean_df.to_csv('word_euclidean_distances.csv', index=False)

# Save shuffled time series to CSV files
shuffled_markov_df = pd.DataFrame(shuffled_series['markov'])
shuffled_philipp_df = pd.DataFrame(shuffled_series['philipp'])
shuffled_random_df = pd.DataFrame(shuffled_series['random'])

shuffled_markov_df.to_csv('Word_mbb.csv', index=False)
shuffled_philipp_df.to_csv('Word_phi.csv', index=False)
shuffled_random_df.to_csv('Word_ran.csv', index=False)

print("Euclidean distances and shuffled time series have been saved to CSV files.")
