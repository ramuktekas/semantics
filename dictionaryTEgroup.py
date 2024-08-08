import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, filtfilt
from statsmodels.tsa.stattools import acf
from sklearn.utils import shuffle
from PyIF import te_compute as te
import csv

# Dictionaries with ACW0 of task for high and low subjects


A1= {'654552': 6, '572045': 5, '552241': 5, '789373': 6, '111514': 6, '135124': 6, '617748': 5, '320826': 5,
          '644246': 5, '130114': 5, '573249': 5, '725751': 6, '467351': 6, '536647': 6, '108323': 5, '833249': 6,
          '765864': 6, '971160': 6, '562345': 6, '943862': 6, '146129': 6, '115017': 6, '966975': 7, '115825': 6,
          '706040': 6, '878877': 5, '132118': 6, '104416': 5, '871762': 6, '114823': 6, '581450': 5, '105923': 6,
          '872764': 5, '751550': 6, '145834': 5, '771354': 5, '140117': 6, '131722': 5, '109123': 7, '782561': 6,
          '818859': 5, '724446': 5, '770352': 5, '878776': 7, '263436': 5, '126426': 5, '995174': 6, '385046': 5,
          '814649': 6, '901442': 6, '128935': 6, '214524': 6, '100610': 5, '134829': 6, '116726': 6, '118225': 6,
          '134627': 6, '690152': 6, '102311': 6, '601127': 6}
A1L= {'572045': 5, '552241': 5, '617748': 5, '320826': 5, '644246': 5, '130114': 5, '573249': 5, '108323': 5,
        '878877': 5, '104416': 5, '581450': 5, '872764': 5, '145834': 5, '771354': 5, '131722': 5, '818859': 5,
        '724446': 5, '770352': 5, '263436': 5, '126426': 5, '385046': 5, '100610': 5, '654552': 6, '789373': 6,
        '111514': 6, '135124': 6, '725751': 6, '467351': 6, '536647': 6, '833249': 6}
A1H= {'765864': 6, '971160': 6, '562345': 6, '943862': 6, '146129': 6, '115017': 6, '115825': 6, '706040': 6,
        '132118': 6, '871762': 6, '114823': 6, '105923': 6, '751550': 6, '140117': 6, '782561': 6, '995174': 6,
        '814649': 6, '901442': 6, '128935': 6, '214524': 6, '134829': 6, '116726': 6, '118225': 6, '134627': 6,
        '690152': 6, '102311': 6, '601127': 6, '966975': 7, '109123': 7, '878776': 7}

PSL= {'654552': 6, '572045': 5, '789373': 7, '111514': 5, '135124': 6, '617748': 5, '320826': 5, '644246': 5,
           '130114': 5, '573249': 5, '725751': 6, '467351': 6, '108323': 5, '833249': 6, '765864': 6,
           '971160': 6, '562345': 6, '943862': 6, '146129': 7, '115017': 6, '966975': 7, '115825': 6, '706040': 6,
           '878877': 5, '132118': 6, '104416': 5, '871762': 6, '114823': 6, '581450': 5, '105923': 6, '872764': 6,
           '751550': 6, '145834': 5, '771354': 5, '140117': 6, '131722': 5, '109123': 7, '782561': 6, '818859': 5,
           '724446': 5, '770352': 5, '878776': 7, '263436': 5, '126426': 5, '995174': 6, '385046': 6, '814649': 6,
           '901442': 7, '128935': 6, '214524': 7, '100610': 5, '134829': 6, '116726': 6, '118225': 6, '134627': 6,
           '690152': 7, '102311': 6, '601127': 7}
PSLL= {'572045': 5, '617748': 5, '320826': 5, '644246': 5, '130114': 5, '573249': 5, '108323': 5, '878877': 5,
         '104416': 5, '581450': 5, '145834': 5, '771354': 5, '131722': 5, '818859': 5, '724446': 5, '770352': 5,
         '263436': 5, '126426': 5, '100610': 5, '111514': 5, '654552': 6, '135124': 6, '725751': 6, '467351': 6,
        '833249': 6, '765864': 6, '971160': 6, '562345': 6,}
PSLH= {'943862': 6, '115017': 6, '115825': 6, '706040': 6, '132118': 6, '871762': 6, '114823': 6,
         '105923': 6, '751550': 6, '140117': 6, '782561': 6, '995174': 6, '385046': 6, '814649': 6, '901442': 7,
         '128935': 6, '214524': 7, '134829': 6, '116726': 6, '118225': 6, '134627': 6, '690152': 7, '102311': 6,
         '601127': 7, '966975': 7, '109123': 7, '878776': 7, '146129': 7}



TA2 = {'654552': 5, '572045': 6, '552241': 6, '789373': 6, '111514': 5, '135124': 5, '617748': 5, '320826': 5,
           '644246': 6, '130114': 5, '573249': 5, '725751': 5, '467351': 6, '536647': 6, '108323': 5, '833249': 6,
           '765864': 6, '971160': 6, '562345': 6, '943862': 6, '146129': 6, '115017': 6, '966975': 7, '115825': 6,
           '706040': 5, '878877': 6, '132118': 6, '104416': 5, '871762': 5, '114823': 5, '581450': 5, '105923': 6,
           '872764': 7, '751550': 6, '145834': 5, '771354': 5, '140117': 5, '131722': 6, '109123': 7, '782561': 7,
           '818859': 5, '724446': 5, '770352': 5, '878776': 6, '263436': 5, '126426': 5, '995174': 6, '385046': 6,
           '814649': 6, '901442': 7, '128935': 5, '214524': 7, '100610': 5, '134829': 6, '116726': 6, '118225': 6,
           '134627': 6, '690152': 6, '102311': 7, '601127': 6}
TA2L= {'654552': 5, '111514': 5, '135124': 5, '617748': 5, '320826': 5, '130114': 5, '573249': 5, '725751': 5,
         '108323': 5, '706040': 5, '104416': 5, '871762': 5, '114823': 5, '581450': 5, '145834': 5, '771354': 5,
         '140117': 5, '818859': 5, '724446': 5, '770352': 5, '263436': 5, '126426': 5, '128935': 5, '100610': 5,
         '572045': 6, '552241': 6, '789373': 6, '644246': 6, '467351': 6, '536647': 6}
TA2H= {'833249': 6, '765864': 6, '971160': 6, '562345': 6, '943862': 6, '146129': 6, '115017': 6, '115825': 6,
         '878877': 6, '132118': 6, '105923': 6, '751550': 6, '131722': 6, '878776': 6, '995174': 6, '385046': 6,
         '814649': 6, '134829': 6, '116726': 6, '118225': 6, '134627': 6, '690152': 6, '601127': 6, '966975': 7,
         '872764': 7, '109123': 7, '782561': 7, '901442': 7, '214524': 7, '102311': 7}

# Define helper functions
def apply_bandpass(signal, lowcut=0.02, highcut=0.1, fs=1.0, order=3):
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

def markov_block_bootstrap(time_series, block_length):
    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]
    shuffled_blocks = shuffle(blocks, random_state=None)
    shuffled_time_series = [item for sublist in shuffled_blocks for item in sublist]
    return shuffled_time_series

def philipp_shuffle(time_series, block_length):
    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]
    shuffled_blocks = [np.random.permutation(block) for block in blocks]
    shuffled_time_series = [item for sublist in shuffled_blocks for item in sublist]
    return shuffled_time_series


def run_analysis(cosine_similarity_path, group_dicts):
    results_dict = {}

    # Load the input time series
    cosine_similarity_ts = pd.read_csv(cosine_similarity_path)['average_wd'].dropna().values

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
        block_lengths = []
        for subject in subjects_dict.keys():
            time_series = fMRI_data[subject].values
            detrended_ts = detrend(time_series)
            filtered_ts = apply_bandpass(detrended_ts)
            processed_bold_signals.append(filtered_ts)
            block_lengths.append(subjects_dict[subject])
            
    
        # Average the BOLD signals for the subgroup
        averaged_bold_signal = np.mean(processed_bold_signals, axis=0)

        # Calculate dynamic ACW time series for the input time series
        input_acw_ts = get_acw_time_series(cosine_similarity_ts)

        # Calculate TE for averaged ACW time series
        averaged_acw_bold = np.mean([get_acw_time_series(ts) for ts in processed_bold_signals], axis=0)
        te_acw_reverse = calculate_te(averaged_acw_bold, input_acw_ts, 3)
        te_acw_forward = calculate_te(input_acw_ts, averaged_acw_bold, 3)

        # Calculate TE for raw time series
        te_raw_reverse = calculate_te(averaged_bold_signal, cosine_similarity_ts, 3)
        te_raw_forward = calculate_te(cosine_similarity_ts, averaged_bold_signal, 3)

        # Create shuffled time series for p-values
        mbb_s = [markov_block_bootstrap(cosine_similarity_ts, 30) for _ in range(1000)]
        phi_s = [philipp_shuffle(cosine_similarity_ts, 30) for _ in range(1000)]
        ran_s = [np.random.permutation(cosine_similarity_ts) for _ in range(1000)]
        mbb_b = [np.mean([markov_block_bootstrap(ts, block_lengths[i]) for i, ts in enumerate(processed_bold_signals)], axis=0) for _ in range(1000)]
        phi_b = [np.mean([philipp_shuffle(ts, block_lengths[i]) for i, ts in enumerate(processed_bold_signals)], axis=0) for _ in range(1000)]
        ran_b = [np.mean([np.random.permutation(ts) for ts in processed_bold_signals], axis=0) for _ in range(1000)]

        # Create shuffled BOLD time series and calculate their ACW time series, then average them
        shuffled_BOLDacw_mbb = [np.mean([get_acw_time_series(markov_block_bootstrap(ts, block_lengths[i])) for i, ts in enumerate(processed_bold_signals)], axis=0) for _ in range(1000)]
        shuffled_BOLDacw_phi = [np.mean([get_acw_time_series(philipp_shuffle(ts, block_lengths[i])) for i, ts in enumerate(processed_bold_signals)], axis=0) for _ in range(1000)]
        shuffled_BOLDacw_ran = [np.mean([get_acw_time_series(np.random.permutation(ts)) for ts in processed_bold_signals], axis=0) for _ in range(1000)]



        # Calculate p-values for ACW TE
        p_values_acw_forward_mbb = p_value(input_acw_ts, shuffled_BOLDacw_mbb, te_acw_forward)
        p_values_acw_reverse_mbb = p_value(averaged_acw_bold, [get_acw_time_series(ts) for ts in mbb_s], te_acw_reverse)
        p_values_acw_forward_phi = p_value(input_acw_ts, shuffled_BOLDacw_phi, te_acw_forward)
        p_values_acw_reverse_phi = p_value(averaged_acw_bold, [get_acw_time_series(ts) for ts in phi_s], te_acw_reverse)
        p_values_acw_forward_ran = p_value(input_acw_ts, shuffled_BOLDacw_ran, te_acw_forward)
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
    'A1_H': A1H,
    'A1_L': A1L,
    'A1all': A1,
    'PSLH': PSLH,
    'PSLL': PSLL,
    'PSLall': PSL,
    'TA2H': TA2H,
    'TA2L': TA2L,
    'TA2all': TA2
}

# Run the analysis
results = run_analysis("adjusted_average_wd_intervals.csv", group_dicts)

# Save results to CSV
with open('te_resultsWDAug8_1423.csv', 'w', newline='') as csvfile:
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
