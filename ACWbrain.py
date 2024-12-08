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
    low_freq = 0.05
    high_freq = 0.4999
    sampling_rate = 1.0
    sos = scipy.signal.butter(N=5, Wn=[low_freq, high_freq], btype='bandpass', fs=sampling_rate, output='sos')
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
cosine_file_path = 'A1_tan_182subs.csv'
worddepth_file_path = 'TA2_tan_182subs.csv'
PSL_file_path = 'PSL_tan_182subs.csv'

cosine_data = pd.read_csv(cosine_file_path)
worddepth_data = pd.read_csv(worddepth_file_path)
PSL_data = pd.read_csv(PSL_file_path)

cosine_similarity_timeseries = cosine_data['A1_tan'].values
word_depth_timeseries = worddepth_data['TA2_tan'].values
PSL_timeseries = PSL_data['PSL_tan'].values


## Calculate dACW time series
#cosine_dacw = get_acw_time_series(cosine_similarity_timeseries)
#word_depth_dacw = get_acw_time_series(word_depth_timeseries)
#
# Apply smoothing (with different parameters for cosine and word depth)
cosine_dacw_smoothed = smooth_timeseries(cosine_similarity_timeseries, sigma=1.5)
word_depth_dacw_smoothed = smooth_timeseries(word_depth_timeseries, sigma=1.5)
PSL_tan_smoothed = smooth_timeseries(PSL_timeseries, sigma=1.5)

# Define the movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]
rest_intervals = [(start -20, end-20 ) for start, end in rest_intervals]
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

silent_intervals = [(start -20, end-20) for start, end in silent_intervals]

# Directory to save the figures
save_directory = '/Volumes/WD/desktop/Figures8Oct'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# First plot: 2 line plots stacked in 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot Cosine dACW (smoothed)
axs[0].plot(cosine_dacw_smoothed, color='blue', linewidth=1)
#axs[0].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[0].set_ylim(2.5,3.8)
axs[0].set_title('Dynamic ACW-0 of A1 (Movie run)', fontweight='bold')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[0].axvspan(rest[0], rest[1], color='grey', alpha=0.5)
for silence in silent_intervals:
    axs[0].axvspan(silence[0], silence[1], color='pink', alpha=0.5)
# Add horizontal grid
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[0].set_xlim(0, len(cosine_dacw_smoothed))

# Plot Word Depth dACW (smoothed)
axs[1].plot(word_depth_dacw_smoothed, color='green', linewidth=1)

#axs[1].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[1].set_ylim(2.6,3.8)
axs[1].set_title('Dynamic ACW-0 of TA2 (Movie run)', fontweight='bold')

# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[1].axvspan(rest[0], rest[1], color='grey', alpha=0.5)
for silence in silent_intervals:
    axs[1].axvspan(silence[0], silence[1], color='pink', alpha=0.5)
# Add horizontal grid
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].set_xlim(0, len(word_depth_dacw_smoothed))

# Now, for the third plot, use axs[2] instead of axs[1]
axs[2].plot(PSL_tan_smoothed, color='red', linewidth=1)
axs[2].set_xlabel('Windows (1 time step = 1 second)', fontweight='bold')
#axs[2].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[2].set_ylim(2.6, 3.8)
axs[2].set_title('Dynamic ACW-0 of PSL (Movie run)',fontweight='bold')
# Set a single y-axis label for the entire figure

fig.supylabel('Autocorrelation Window-0 (seconds)', fontsize=12, fontweight='bold')
# Add rest intervals as grey boxes
for rest in rest_intervals:
    axs[2].axvspan(rest[0], rest[1], color='grey', alpha=0.5)
for silence in silent_intervals:
    axs[2].axvspan(silence[0], silence[1], color='pink', alpha=0.5)
# Add horizontal grid for the third plot
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

axs[2].set_xlim(0, len(PSL_tan_smoothed))

# Save and show the line plots
save_path_line_plots = os.path.join(save_directory, 'DY_ACW_MOVIE.png')
plt.tight_layout()
plt.savefig(save_path_line_plots, bbox_inches='tight')
plt.show()

print(f"Line plot figure saved at: {save_path_line_plots}")
#import matplotlib.pyplot as plt
#
## Pearson correlation values and labels
#correlations = [0.854, 0.467, 0.353]
#labels = ['A1-TA2', 'PSL-TA2', 'A1-PSL']
#
## Define colors
#colors = ['orange', 'purple', 'black']
#
## Create a thinner and tall bar plot
#fig, ax = plt.subplots(figsize=(2, 8))  # Reduced width for thinner plot
#
## Plot the Pearson correlation values with full-width bars for no spacing
#ax.bar(range(len(correlations)), correlations, color=colors, width=1.0)  # Full-width bars
#
## Set y-axis limit and labels on the right side
#ax.set_ylim(0, 1)
#
## Adjust title position by setting 'loc' to 'left' and fine-tuning pad
#ax.set_title('Pearson Correlation', pad=20, fontweight='bold', loc='left')  # Move title slightly to the right
#
## Set ticks on the right side of the plot
#ax.yaxis.tick_right()
#
## Adjust x-axis limits tightly around the bars to remove gaps
#ax.set_xlim(-0.5, len(correlations) - 0.5)  # Tighten x-axis limits based on number of bars
#ax.set_xticks(range(len(correlations)))
#ax.set_xticklabels(labels, rotation=90, fontsize=8, fontweight='bold')
#
## Remove any padding between the axes and the bars
#ax.margins(x=0)
#
## Use tight layout to ensure no cutoff
#plt.tight_layout()
#
## Add horizontal grid and set its transparency
#ax.yaxis.grid(True, linestyle='--', alpha=0.5)
#
## Save and show the bar plot
#save_path = 'dycor_movie.png'
#plt.savefig(save_path, bbox_inches='tight')
#plt.show()



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
cosine_file_path = 'A1_rantest.csv'
worddepth_file_path = 'TA2_rantest.csv'
PSL_file_path = 'PSL_rantest.csv'

cosine_data = pd.read_csv(cosine_file_path)
worddepth_data = pd.read_csv(worddepth_file_path)
PSL_data = pd.read_csv(PSL_file_path)

cosine_similarity_timeseries = cosine_data['A1_rpn'].values
word_depth_timeseries = worddepth_data['TA2_rpn'].values
PSL_timeseries = PSL_data['PSL_rpn'].values


## Calculate dACW time series
#cosine_dacw = get_acw_time_series(cosine_similarity_timeseries)
#word_depth_dacw = get_acw_time_series(word_depth_timeseries)
#
# Apply smoothing (with different parameters for cosine and word depth)
cosine_dacw_smoothed = smooth_timeseries(cosine_similarity_timeseries, sigma=0.8)
word_depth_dacw_smoothed = smooth_timeseries(word_depth_timeseries, sigma=0.8)
PSL_tan_smoothed = smooth_timeseries(PSL_timeseries, sigma=0.8)

# Define the movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]

# Directory to save the figures
save_directory = '/Volumes/WD/desktop/Figures8Oct'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# First plot: 2 line plots stacked in 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot Cosine dACW (smoothed)
axs[0].plot(cosine_dacw_smoothed, color='blue', linewidth=1)
#axs[0].set_ylabel('Autocorrelation Window-0 (ACW-0)')
axs[0].set_ylim(5.2, 6.5)
axs[0].set_title('Dynamic ACW-0 of A1 (Rest)', fontweight='bold')

## Add rest intervals as grey boxes
#for rest in rest_intervals:
#    axs[0].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[0].set_xlim(0, len(cosine_dacw_smoothed))

# Plot Word Depth dACW (smoothed)
axs[1].plot(word_depth_dacw_smoothed, color='green', linewidth=1)

#axs[1].set_ylabel('Autocorrelation Window-0 (ACW-0)', fontweight='bold')
axs[1].set_ylim(5.4, 6.5)
axs[1].set_title('Dynamic ACW-0 of TA2 (Rest)', fontweight='bold')

## Add rest intervals as grey boxes
#for rest in rest_intervals:
#    axs[1].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].set_xlim(0, len(word_depth_dacw_smoothed))

# Now, for the third plot, use axs[2] instead of axs[1]
axs[2].plot(PSL_tan_smoothed, color='red', linewidth=1)
axs[2].set_xlabel('Windows (1 time step = 1 second)', fontweight='bold')
#axs[2].set_ylabel('Autocorrelation Window-0 (ACW-0)', fontweight='bold')
axs[2].set_ylim(5.2, 6.5)
axs[2].set_title('Dynamic ACW of PSL (Rest)', fontweight='bold')
fig.supylabel('Autocorrelation Window-0 (ACW-0)', fontsize=12, fontweight='bold')

## Add rest intervals as grey boxes
#for rest in rest_intervals:
#    axs[2].axvspan(rest[0], rest[1], color='grey', alpha=0.5)

# Add horizontal grid for the third plot
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

axs[2].set_xlim(0, len(PSL_tan_smoothed))

# Save and show the line plots
save_path_line_plots = os.path.join(save_directory, 'dy_rest.png')
plt.tight_layout()
#plt.savefig(save_path_line_plots, bbox_inches='tight')
plt.show()

print(f"Line plot figure saved at: {save_path_line_plots}")
#from scipy.stats import pearsonr
#
## Compute pairwise Pearson correlations
#corr_A1_TA2, _ = pearsonr(cosine_dacw_smoothed, word_depth_dacw_smoothed)
#corr_A1_PSL, _ = pearsonr(cosine_dacw_smoothed, PSL_tan_smoothed)
#corr_TA2_PSL, _ = pearsonr(word_depth_dacw_smoothed, PSL_tan_smoothed)
#
## Print the correlation values
#print(f"Pearson correlation between A1 and TA2: {corr_A1_TA2:.4f}")
#print(f"Pearson correlation between A1 and PSL: {corr_A1_PSL:.4f}")
#print(f"Pearson correlation between TA2 and PSL: {corr_TA2_PSL:.4f}")
#import matplotlib.pyplot as plt
#
## Pearson correlation values and labels
#correlations = [0.4162, 0.4711, 0.5705]
#labels = ['A1-TA2', 'PSL-TA2', 'A1-PSL']
#
## Define colors
#colors = ['orange', 'purple', 'black']
#
## Create a thinner and tall bar plot
#fig, ax = plt.subplots(figsize=(2, 8))  # Reduced width for thinner plot
#
## Plot the Pearson correlation values with full-width bars for no spacing
#ax.bar(range(len(correlations)), correlations, color=colors, width=1.0)  # Full-width bars
#
## Set y-axis limit and labels on the right side
#ax.set_ylim(0, 1)
#
## Adjust title position by setting 'loc' to 'left' and fine-tuning pad
#ax.set_title('Pearson Correlation', pad=20, fontweight='bold', loc='left')  # Move title slightly to the right
#
## Set ticks on the right side of the plot
#ax.yaxis.tick_right()
#
## Adjust x-axis limits tightly around the bars to remove gaps
#ax.set_xlim(-0.5, len(correlations) - 0.5)  # Tighten x-axis limits based on number of bars
#ax.set_xticks(range(len(correlations)))
#ax.set_xticklabels(labels, rotation=90, fontsize=8, fontweight='bold')
#
## Remove any padding between the axes and the bars
#ax.margins(x=0)
#
## Use tight layout to ensure no cutoff
#plt.tight_layout()
#
## Add horizontal grid and set its transparency
#ax.yaxis.grid(True, linestyle='--', alpha=0.5)
#
## Save and show the bar plot
#save_path = 'dycorr_rest.png'
##plt.savefig(save_path, bbox_inches='tight')
#plt.show()
#
#print(f"Bar plot figure saved at: {save_path}")
