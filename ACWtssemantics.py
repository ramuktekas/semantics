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

def get_acw_time_series(time_series, window_size=600, step_size=10):
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
cosine_file_path = '1_0-1_7T_MOVIE2_HOI.csv'
worddepth_file_path = 'WD_MOVIE2_HOI.csv'

cosine_data = pd.read_csv(cosine_file_path)
worddepth_data = pd.read_csv(worddepth_file_path)

cosine_similarity_timeseries = cosine_data['Cosine_Similarity'].values[200:-200]
word_depth_timeseries = worddepth_data['Word_depth'].values[200:-200]

# Compute averages for cosine similarity timeseries
#cosine_similarity_timeseries= np.mean(
#    cosine_similarity_timeseries[:len(cosine_similarity_timeseries) // 10 * 10].reshape(-1, 10), axis=1
#)
#
## Compute averages for word depth timeseries
#word_depth_timeseries= np.mean(
#    word_depth_timeseries[:len(word_depth_timeseries) // 10 * 10].reshape(-1, 10), axis=1
#)
# Calculate dACW time series
cosine_dacw = get_acw_time_series(cosine_similarity_timeseries)
word_depth_dacw = get_acw_time_series(word_depth_timeseries)
#cosine_dacw = cosine_similarity_timeseries
#word_depth_dacw = word_depth_timeseries

# Apply smoothing (with different parameters for cosine and word depth)
cosine_dacw_smoothed = smooth_timeseries(cosine_dacw, sigma=1.5)
word_depth_dacw_smoothed = smooth_timeseries(word_depth_dacw, sigma=1.5)

# Define the movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
#rest_intervals = [(0, 200), (2460, 2670), (5250, 5450), (7940, 8150), (8970, 9000)]
rest_intervals = [(246, 267), (525, 545), (794, 815)]

# Multiply each number in the tuples by 10
#rest_intervals = [(start * 10, end * 10) for start, end in rest_intervals]
silent_intervals = [
    (24.6, 25.0), (25.9, 26.0), (27.9, 28.1), (31.7, 31.8), (34.5, 34.7), (40.5, 41.2),
    (43.7, 43.8), (45.9, 46.0), (49.1, 49.5), (49.9, 50.5), (52.2, 52.3), (52.7, 53.5),
    (54.6, 55.0), (59.3, 60.2), (63.7, 106.0), (106.9, 108.7), (109.5, 134.1),
    (140.7, 140.8), (142.9, 143.8), (151.8, 153.4), (154.0, 159.1), (162.7, 167.5),
    (172.5, 226.0), (227.1, 239.0), (243.6, 243.9), (267.0, 289.3), (290.7, 292.2),
    (298.7, 300.0), (300.6, 301.2), (303.0, 303.3), (303.5, 303.6), (304.1, 312.3),
    (313.1, 313.6), (313.7, 313.8), (315.1, 315.3), (323.7, 324.8), (325.0, 325.1),
    (326.8, 327.0), (333.0, 333.6), (336.4, 336.8), (338.0, 338.1), (338.6, 339.5),
    (345.7, 345.8), (346.7, 347.6), (350.1, 351.6), (352.3, 352.8), (353.5, 354.3),
    (355.2, 356.6), (358.0, 361.5), (364.2, 364.3), (367.7, 367.8), (369.5, 369.6),
    (371.3, 372.1), (374.7, 374.8), (380.6, 380.8), (381.5, 382.0), (386.9, 387.0),
    (391.1, 392.1), (392.7, 394.1), (395.0, 395.8), (397.8, 398.0), (404.9, 406.1),
    (408.4, 408.5), (411.1, 413.8), (414.2, 415.0), (415.3, 416.3), (418.1, 419.1),
    (424.5, 424.6), (425.2, 425.6), (426.1, 427.1), (438.5, 438.8), (439.8, 440.0),
    (444.3, 450.1), (450.5, 451.0), (454.0, 454.6), (460.0, 460.6), (464.9, 465.8),
    (466.0, 466.6), (470.8, 471.6), (473.4, 474.6), (476.3, 476.6), (483.8, 484.0),
    (491.2, 492.0), (494.8, 495.3), (498.3, 498.6), (501.7, 504.3), (505.7, 506.6),
    (514.1, 516.7), (520.6, 525.0), (545.0, 545.9), (547.5, 548.2), (550.7, 551.0),
    (552.3, 553.2), (554.5, 554.6), (557.7, 558.2), (565.0, 565.7), (569.5, 570.7),
    (571.7, 581.9), (582.5, 582.6), (583.6, 585.5), (586.8, 587.6), (587.7, 589.2),
    (590.1, 590.2), (591.5, 592.2), (593.7, 593.9), (595.0, 605.7), (609.1, 609.2),
    (612.8, 612.9), (616.4, 617.0), (617.3, 619.6), (629.8, 630.7), (631.8, 632.2),
    (633.5, 634.0), (635.6, 637.5), (637.7, 639.6), (644.0, 645.0), (656.4, 656.5),
    (662.9, 663.0), (672.0, 672.1), (679.4, 679.5), (680.4, 680.5), (686.6, 686.7),
    (687.6, 687.7), (691.0, 691.1), (692.4, 692.5), (698.0, 699.2), (700.9, 704.2),
    (705.5, 705.7), (713.9, 714.5), (714.7, 715.3), (722.3, 723.0), (731.1, 731.2),
    (734.1, 734.3), (734.5, 734.7), (736.7, 738.0), (740.0, 741.8), (744.8, 745.6),
    (746.3, 747.7), (748.5, 749.1), (750.4, 751.0), (752.6, 752.8), (757.9, 758.0),
    (759.6, 760.1), (763.5, 764.2), (766.3, 767.2), (777.5, 782.3), (783.6, 786.2),
    (786.4, 788.2), (788.6, 794.0), (815.0, 854.7), (858.6, 858.7), (860.8, 861.1),
    (862.0, 877.2), (879.5, 879.7), (881.6, 897.0)
]

#silent_intervals = [(start * 10, end * 10) for start, end in silent_intervals]

# Directory to save the figure
save_directory = '/Volumes/WD/desktop/Figures8Oct'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Create subplots: one for Cosine dACW and one for Word Depth dACW
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Cosine dACW (smoothed)
axs[0].plot(cosine_dacw_smoothed/10, color='blue', linewidth=1)
#axs[0].plot(cosine_dacw_smoothed, color='blue', linewidth=1)
#axs[0].set_xlabel('Windows (1 time step = 1 second)', fontweight = 'bold')

#axs[0].set_ylabel('Cosine Similarity', fontweight = 'bold')
#axs[0].set_ylim(0, 50)
axs[0].set_title('Dyanmic ACW-0 of Sentence Similarity', fontweight = 'bold')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[0].axvspan(rest[0], rest[1], color='grey', alpha=0.5)
# Add silence intervals in pink boxes
for silence in silent_intervals:
    axs[0].axvspan(silence[0], silence[1], color='pink', alpha=0.5)
# Set y-axis and origin at the origin
axs[0].spines['left'].set_position(('data', 0))
axs[0].set_xlim(0, len(cosine_dacw_smoothed))
#axs[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x / 10)}'))

# Plot Word Depth dACW (smoothed)
#axs[1].plot(word_depth_dacw_smoothed, color='green', linewidth=1)
axs[1].plot(word_depth_dacw_smoothed/10, color='green', linewidth=1)
axs[1].set_xlabel('Windows (step size= 1 second)', fontweight = 'bold')
#axs[1].set_ylabel('Dynamic ACW-0 of Word Depths', fontweight = 'bold')
#axs[1].set_ylim(0, 50)
axs[1].set_title('Dynamic ACW-0 of Word Depths', fontweight = 'bold')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[1].axvspan(rest[0], rest[1], color='grey', alpha=0.5)
for silence in silent_intervals:
    axs[1].axvspan(silence[0], silence[1], color='pink', alpha=0.5)

# Set y-axis and origin at the origin
axs[1].spines['left'].set_position(('data', 0))
axs[1].set_xlim(0, len(word_depth_dacw_smoothed))
axs[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x / 10)}'))
fig.text(0.04, 0.5, 'Autocorrelation Window-0 (ACW-0)', va='center', rotation='vertical', fontweight='bold')


# Save and show the plot
save_path = os.path.join(save_directory, 'DY_ACW0_semantics.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Figure saved at: {save_path}")

