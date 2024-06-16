import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, filtfilt
from statsmodels.tsa.stattools import acf
from sklearn.utils import shuffle
from PyIF import te_compute as te
import csv

# Dictionaries with ACW0 of task for high and low subjects

# A1 Region
a1_H = {
    '724446': 5, '878776': 22, '573249': 5, '263436': 5, '943862': 14, '995174': 24,
    '115825': 7, '140117': 8, '562345': 6, '789373': 8, '115017': 5, '134627': 15,
    '105923': 13, '128935': 7, '132118': 11, '135124': 14, '966975': 6, '111514': 13,
    '901442': 7, '871762': 13, '770352': 9, '109123': 6, '782561': 16, '320826': 14,
    '467351': 6, '214524': 5, '116726': 9, '146129': 14, '118225': 17
}

a1_L = {
    '104416': 6, '552241': 17, '690152': 11, '725751': 12, '536647': 11, '131722': 7,
    '771354': 13, '617748': 6, '114823': 10, '145834': 5, '102311': 7, '385046': 8,
    '818859': 8, '971160': 9, '108323': 5, '130114': 8, '765864': 6, '814649': 15,
    '126426': 25, '751550': 10, '100610': 21, '572045': 6, '654552': 5, '833249': 7,
    '706040': 7, '581450': 14, '134829': 9, '878877': 4, '644246': 14, '872764': 9
}

# PSL Region
pslH = {
    '943862': 13, '770352': 22, '573249': 29, '115017': 13, '878877': 5, '814649': 14,
    '872764': 7, '725751': 12, '128935': 6, '105923': 8, '467351': 12, '833249': 15,
    '100610': 8, '102311': 31, '995174': 19, '146129': 8, '116726': 5, '878776': 25,
    '385046': 19, '562345': 6, '263436': 5, '108323': 13, '132118': 20, '118225': 4,
    '690152': 10, '966975': 6, '214524': 19
}

pslL = {
    '130114': 13, '581450': 6, '771354': 14, '131722': 21, '782561': 19, '115825': 8,
    '134829': 13, '901442': 8, '617748': 9, '724446': 15, '971160': 8, '654552': 15,
    '126426': 13, '644246': 9, '818859': 9, '134627': 16, '789373': 12, '871762': 14,
    '706040': 18, '751550': 6, '104416': 8, '135124': 10, '140117': 10, '111514': 6,
    '145834': 14, '572045': 29, '320826': 7, '114823': 8, '109123': 22, '765864': 7
}

# TA2 Region
ta2H = {
    '116726': 22, '878776': 13, '128935': 12, '562345': 10, '690152': 8, '943862': 14,
    '751550': 14, '536647': 16, '871762': 13, '131722': 25, '706040': 8, '102311': 10,
    '105923': 13, '263436': 22, '135124': 11, '971160': 5, '146129': 13, '115017': 5,
    '901442': 6, '100610': 16, '814649': 9, '572045': 8, '109123': 6, '782561': 15,
    '966975': 4, '118225': 11, '878877': 4, '467351': 5, '872764': 9
}

ta2L = {
    '320826': 12, '725751': 10, '617748': 8, '114823': 13, '134627': 12, '724446': 13,
    '111514': 12, '995174': 20, '115825': 16, '126426': 24, '130114': 16, '833249': 8,
    '145834': 5, '134829': 10, '654552': 7, '104416': 14, '818859': 10, '552241': 16,
    '770352': 14, '214524': 7, '771354': 11, '385046': 10, '765864': 15, '789373': 12,
    '644246': 13, '132118': 15, '140117': 12, '573249': 7, '108323': 13, '581450': 14
}

# Define helper functions
def apply_bandpass(signal, lowcut=0.02, highcut=0.2, fs=1.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def calculate_acw(ts):
    acw_func = acf(ts, nlags=len(ts)-1, fft=True)
    acw_0 = np.argmax(acw_func <= 0)
    return acw_0

def get_acw_time_series(time_series, window_size=150, step_size=1):
    detrended_ts = detrend(time_series)
    filtered_ts = apply_bandpass(detrended_ts)
    acw_series = [calculate_acw(filtered_ts[i:i+window_size]) for i in range(0, len(filtered_ts) - window_size + 1, step_size)]
    return acw_series

def calculate_te(source, target, m):
    min_length = min(len(source), len(target))
    source = np.array(source)[:min_length]
    target = np.array(target)[:min_length]
    tau_values = np.arange(1, 11)
    te_results = [te.te_compute(source, target, m, tau) for tau in tau_values]
    return te_results

def p_value(input_ts, shuffle_ts, te_values):
    greater_count = {tau: 0 for tau in range(len(te_values))}
    for ts in shuffle_ts:
        shuffle_te = calculate_te(input_ts, ts, 3)
        for tau, te_value in enumerate(shuffle_te):
            if te_value >= te_values[tau]:
                greater_count[tau] += 1
    p_values = {tau: greater_count[tau] / len(shuffle_ts) for tau in range(len(te_values))}
    return p_values

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

def run_analysis(cosine_similarity_path, group_dicts):
    results_dict = {}

    # Load the input time series
    cosine_similarity_ts = pd.read_csv(cosine_similarity_path)['Cosine Similarity'].dropna().values

    for group_name, subjects_dict in group_dicts.items():
        # Extract the ROI and subgroup
        roi = group_name[:3]
        subgroup = group_name[3:]

        # Load the corresponding fMRI task data
        if roi == 'A1_':
            fMRI_data = pd.read_csv('A1_tpn.csv')
        elif roi == 'PSL':
            fMRI_data = pd.read_csv('PSL_tpn.csv').replace(',', '', regex=True).astype(float)
        elif roi == 'TA2':
            fMRI_data = pd.read_csv('TA2_tpn.csv')

        # Detrend and bandpass fMRI time series for each subject
        processed_bold_signals = []
        for subject in subjects_dict.keys():
            time_series = fMRI_data[subject].values
            detrended_ts = detrend(time_series)
            filtered_ts = apply_bandpass(detrended_ts)
            processed_bold_signals.append(filtered_ts)

        # Average the BOLD signals for the subgroup
        averaged_bold_signal = np.mean(processed_bold_signals, axis=0)

        # Calculate dynamic ACW time series for the input time series
        input_acw_ts = get_acw_time_series(cosine_similarity_ts)

        # Calculate TE for averaged ACW time series
        averaged_acw_bold = get_acw_time_series(averaged_bold_signal)
        te_acw_reverse = calculate_te(averaged_acw_bold, input_acw_ts, 3)
        te_acw_forward = calculate_te(input_acw_ts, averaged_acw_bold, 3)

        # Calculate TE for raw time series
        te_raw_reverse = calculate_te(averaged_bold_signal, cosine_similarity_ts, 3)
        te_raw_forward = calculate_te(cosine_similarity_ts, averaged_bold_signal, 3)

        # Create shuffled time series for p-values
        mbb_s = [markov_block_bootstrap(cosine_similarity_ts) for _ in range(1000)]
        phi_s = [philipp_shuffle(cosine_similarity_ts) for _ in range(1000)]
        ran_s = [np.random.permutation(cosine_similarity_ts) for _ in range(1000)]
        mbb_b = [markov_block_bootstrap(averaged_bold_signal) for _ in range(1000)]
        phi_b = [philipp_shuffle(averaged_bold_signal) for _ in range(1000)]
        ran_b = [np.random.permutation(averaged_bold_signal) for _ in range(1000)]

        # Calculate p-values for ACW TE
        p_values_acw_forward_mbb = p_value(input_acw_ts, [get_acw_time_series(ts) for ts in mbb_b], te_acw_forward)
        p_values_acw_reverse_mbb = p_value(averaged_acw_bold, [get_acw_time_series(ts) for ts in mbb_s], te_acw_reverse)
        p_values_acw_forward_phi = p_value(input_acw_ts, [get_acw_time_series(ts) for ts in phi_b], te_acw_forward)
        p_values_acw_reverse_phi = p_value(averaged_acw_bold, [get_acw_time_series(ts) for ts in phi_s], te_acw_reverse)
        p_values_acw_forward_ran = p_value(input_acw_ts, [get_acw_time_series(ts) for ts in ran_b], te_acw_forward)
        p_values_acw_reverse_ran = p_value(averaged_acw_bold, [get_acw_time_series(ts) for ts in ran_s], te_acw_reverse)

        # Calculate p-values for raw TE
        p_values_raw_forward_mbb = p_value(cosine_similarity_ts, mbb_b, te_raw_forward)
        p_values_raw_forward_phi = p_value(cosine_similarity_ts, phi_b, te_raw_forward)
        p_values_raw_forward_ran = p_value(cosine_similarity_ts, ran_b, te_raw_forward)
        p_values_raw_reverse_mbb = p_value(averaged_bold_signal, mbb_s, te_raw_reverse)
        p_values_raw_reverse_phi = p_value(averaged_bold_signal, phi_s, te_raw_reverse)
        p_values_raw_reverse_ran = p_value(averaged_bold_signal, ran_s, te_raw_reverse)

        # Store the results
        results_dict[group_name] = {
            'te_acw_forward': te_acw_forward,
            'te_acw_reverse': te_acw_reverse,
            'te_raw_forward': te_raw_forward,
            'te_raw_reverse': te_raw_reverse,
            'p_af_mbb': p_values_acw_forward_mbb,
            'p_ar_mbb': p_values_acw_reverse_mbb,
            'p_af_phi': p_values_acw_forward_phi,
            'p_ar_phi': p_values_acw_reverse_phi,
            'p_af_ran': p_values_acw_forward_ran,
            'p_ar_ran': p_values_acw_reverse_ran,
            'p_rf_mbb': p_values_raw_forward_mbb,
            'p_rr_mbb': p_values_raw_reverse_mbb,
            'p_rf_phi': p_values_raw_forward_phi,
            'p_rr_phi': p_values_raw_reverse_phi,
            'p_rf_ran': p_values_raw_forward_ran,
            'p_rr_ran': p_values_raw_reverse_ran
        }

    return results_dict

# Define the group dictionaries as provided
group_dicts = {
    'A1_H': a1_H,
    'A1_L': a1_L,
    'PSLH': pslH,
    'PSLL': pslL,
    'TA2H': ta2H,
    'TA2L': ta2L
}

# Run the analysis
results = run_analysis("s3-7T_MOVIE2_HO1.csv", group_dicts)

# Save results to CSV
with open('te_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['group', 'te_acw_forward', 'te_acw_reverse', 'te_raw_forward', 'te_raw_reverse',
                  'p_af_mbb', 'p_ar_mbb', 'p_af_phi', 'p_ar_phi', 'p_af_ran', 'p_ar_ran',
                  'p_rf_mbb', 'p_rr_mbb', 'p_rf_phi', 'p_rr_phi', 'p_rf_ran', 'p_rr_ran']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for group, result in results.items():
        row = {'group': group}
        for key, value in result.items():
            row[key] = value
        writer.writerow(row)
