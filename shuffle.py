import numpy as np
from statsmodels.tsa.stattools import acf
from sklearn.utils import shuffle
import random
'''
these algorithms are the shuffling methods I use commonly. 
Random shuffling destroys internal structure of a signal that you may not always want for your analysis
instead, these two methods selectively alter the signal structure that can be better used for comparison
Compartmentalise the timeseries into blocks of length equal to the Autocorrelation of the window of the timeseries
(know more about them here: https://www.georgnorthoff.com/s/acw.m)
Markov Block bootstrap shuffled the blocks, 
while philipp's shuffle inside the block
7-6-24
'''
def markov_block_bootstrap(time_series, sampling_rate=1.0):
    # Calculate block length using the first zero-crossing of the autocorrelation function
    acw_func = acf(time_series, nlags=len(time_series) - 1, fft=True)
    block_length = np.argmax(acw_func <= 0) / sampling_rate

    # Ensure block length is at least 1
    block_length = max(int(block_length), 1)

    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]

    # Shuffle the blocks
    shuffled_blocks = shuffle(blocks, random_state=None)

    # Flatten the list of shuffled blocks into a single time series
    shuffled_time_series = [item for sublist in shuffled_blocks for item in sublist]
    return shuffled_time_series


def philipp_shuffle(time_series, sampling_rate=1.0):
    # Calculate block length using the first zero-crossing of the autocorrelation function
    acw_func = acf(time_series, nlags=len(time_series) - 1, fft=True)
    block_length = np.argmax(acw_func <= 0) / sampling_rate

    # Ensure block length is at least 1
    block_length = max(int(block_length), 1)

    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]

    # Shuffle within each block
    shuffled_blocks = [np.random.permutation(block) for block in blocks]

    # Flatten the list of shuffled blocks into a single time series
    shuffled_time_series = [item for sublist in shuffled_blocks for item in sublist]
    return shuffled_time_series
