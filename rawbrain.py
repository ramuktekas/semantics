import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to process each file and return the mean timeseries
# Function to process each file and return the mean timeseries
def process_file(filename):
    df = pd.read_csv(filename)
    bold_signals = []
    
    for col in df.columns[1:]:  # Ignore the first column (subject ID)
        timeseries = df[col].values[20:-20]  # Remove first and last 20 values
        detrended = detrend(timeseries)  # Detrend the time series
        filtered = bandpass_filter(detrended, 0.05, 0.4999, fs=1, order=5)  # Bandpass filter
        
        # Apply Gaussian smoothing with a width of 1.5
        smoothed = gaussian_filter1d(filtered, sigma=0.5)  # Smooth with a Gaussian kernel
        
        bold_signals.append(smoothed)
    
    # Take the mean across all subjects for each timepoint
    mean_bold = np.mean(bold_signals, axis=0)
    return mean_bold

# Load and process files
mean_A1 = process_file("A1_rpn_177subs.csv")
mean_TA2 = process_file("TA2_rpn_177subs.csv")
mean_PSL = process_file("PSL_rpn_177subs.csv")

# Movie and rest intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]
rest_intervals = [(0, 20), (246, 267), (525, 545), (794, 815), (897, 900)]
rest_intervals = [(0,1000)]
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
# Generate a time axis (assuming 900 time points in each timeseries)
time = np.arange(len(mean_A1))  # X axis goes from 0 to 900 seconds

# Plot setup with 3 rows
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

# Plotting A1 BOLD signal
axs[0].plot(time, mean_A1, color='blue')
axs[0].set_title('BOLD Signal A1 (Rest)',fontweight='bold')
#axs[0].set_ylabel('BOLD Signal')

# Plotting TA2 BOLD signal
axs[1].plot(time, mean_TA2, color='green')
axs[1].set_title('BOLD Signal TA2 (Rest)',fontweight='bold')
#axs[1].set_ylabel('BOLD Signal')

# Plotting PSL BOLD signal
axs[2].plot(time, mean_PSL[:900], color='red')
axs[2].set_title('BOLD Signal PSL (Rest)',fontweight='bold')
#axs[2].set_ylabel('BOLD Signal')
axs[2].set_xlabel('Time (seconds)',fontweight='bold')
fig.supylabel('BOLD signal', fontsize=12, fontweight='bold')
# Add grey boxes for rest intervals on all plots
for ax in axs:
    for rest in rest_intervals:
        ax.axvspan(rest[0], rest[1], color='grey', alpha=0.3)
#    for silence in silent_intervals:
#        ax.axvspan(silence[0], silence[1], color='pink', alpha=0.5)

# Set Y-axis limits to be the same across all plots
max_y = max(mean_A1.max(), mean_TA2.max(), mean_PSL.max())
min_y = min(mean_A1.min(), mean_TA2.min(), mean_PSL.min())
for ax in axs:
    ax.set_ylim([-101, 101])
#print(min_y,max_y)
# Add horizontal grid to all plots
for ax in axs:
    ax.yaxis.grid(True, linestyle='--', alpha=1)

# Limit X-axis to 0-900 seconds
plt.xlim(0, len(mean_A1))

# Save the first plot as 'Braintask.png'
plt.tight_layout()
braintask_save_path = 'BOLD_REST.png'
plt.savefig(braintask_save_path, bbox_inches='tight')

# Display plot
plt.show()
