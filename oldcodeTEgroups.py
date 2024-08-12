import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from PyIF import te_compute
from scipy.signal import correlate, butter, filtfilt
from scipy.stats import zscore
import scipy.signal
from statsmodels.tsa.stattools import acf

# Load the time series data
A1_tpn_df = pd.read_csv('A1_tpn.csv').drop(columns=['A1_tpn'])
# Load the Cosine Similarity time series and remove NaN values
cosine_similarity_ts = pd.read_csv("adjusted_average_wd_intervals.csv")['average_wd']
cosine_similarity_ts = cosine_similarity_ts.dropna().values
# Extract the Cosine Similarity time series
cosine_ts = cosine_similarity_ts

#cosine_similarity_ts = pd.read_csv("/Users/satrokommos/Documents/9th sem/code/data/Text/s3-7T_MOVIE2_HO1.csv")['Cosine Similarity']
#cosine_similarity_ts = cosine_similarity_ts.dropna().values
'''#TA2 acw rest
acw_values = {
    '654552': 5.363177805800756,
    '572045': 5.484810126582278,
    '552241': 6.230769230769231,
    '789373': 6.140328697850822,
    '111514': 5.2735368956743,
    '135124': 5.452711223203027,
    '617748': 5.354350567465321,
    '320826': 5.680555555555555,
    '644246': 5.4936868686868685,
    '130114': 5.620915032679738,
    '573249': 5.19672131147541,
    '725751': 5.479746835443038,
    '467351': 5.140684410646388,
    '536647': 6.121212121212121,
    '108323': 5.564943253467844,
    '833249': 6.508196721311475,
    '765864': 5.768939393939394,
    '971160': 5.539141414141414,
    '562345': 5.416141235813367,
    '943862': 5.971902937420179,
    '146129': 6.230271668822769,
    '115017': 5.529634300126103,
    '966975': 6.5,
    '115825': 5.783715012722646,
    '706040': 5.269861286254729,
    '878877': 5.079646017699115,
    '132118': 5.416141235813367,
    '104416': 5.857323232323233,
    '871762': 5.472222222222222,
    '114823': 5.761363636363637,
    '581450': 5.701799485861183,
    '105923': 6.015170670037927,
    '872764': 5.648171500630517,
    '751550': 5.773417721518987,
    '145834': 5.337547408343869,
    '771354': 5.566204287515763,
    '140117': 5.376903553299492,
    '131722': 5.1856060606060606,
    '109123': 6.015523932729625,
    '782561': 5.597977243994943,
    '818859': 5.204834605597965,
    '724446': 5.262086513994911,
    '770352': 5.287760416666667,
    '878776': 6.03530895334174,
    '263436': 5.6313131313131315,
    '126426': 5.325346784363178,
    '995174': 6.132315521628499,
    '385046': 5.235443037974684,
    '814649': 5.300884955752212,
    '901442': 5.9785082174462705,
    '128935': 5.31130876747141,
    '214524': 5.594904458598726,
    '100610': 5.340909090909091,
    '134829': 5.634980988593156,
    '116726': 5.943253467843632,
    '118225': 6.308953341740227,
    '134627': 6.086624203821656,
    '690152': 5.624525916561315,
    '102311': 5.8063291139240505,
    '601127': 5.542351453855878
}'''


#A1 ACW rest values
acw_values = {
    '654552': 5.265588914549654,
    '572045': 5.351039260969977,
    '552241': 6.517321016166282,
    '789373': 5.672055427251732,
    '111514': 5.10392609699769,
    '135124': 5.212471131639723,
    '617748': 5.02540415704388,
    '320826': 5.575057736720554,
    '644246': 5.02540415704388,
    '130114': 5.344110854503464,
    '573249': 5.041570438799076,
    '725751': 5.196304849884527,
    '467351': 5.157043879907621,
    '536647': 5.727482678983834,
    '108323': 5.302540415704388,
    '833249': 6.720554272517321,
    '765864': 5.586605080831409,
    '971160': 5.3140877598152425,
    '562345': 5.327944572748268,
    '943862': 6.02540415704388,
    '146129': 5.621247113163972,
    '115017': 5.413394919168591,
    '966975': 6.7829099307159355,
    '115825': 6.122401847575058,
    '706040': 5.40877598152425,
    '878877': 5.064665127020786,
    '132118': 5.1270207852194,
    '104416': 5.387990762124711,
    '871762': 5.401847575057737,
    '114823': 5.690531177829099,
    '581450': 5.450346420323326,
    '105923': 6.0392609699769055,
    '872764': 5.337182448036952,
    '751550': 5.803695150115473,
    '145834': 5.23094688221709,
    '771354': 5.113163972286374,
    '140117': 5.173210161662817,
    '131722': 5.05080831408776,
    '109123': 6.106235565819861,
    '782561': 5.445727482678984,
    '818859': 5.064665127020786,
    '724446': 5.12933025404157,
    '770352': 5.1362586605080836,
    '878776': 6.882217090069284,
    '263436': 5.427251732101617,
    '126426': 5.1270207852194,
    '995174': 5.849884526558891,
    '385046': 5.117782909930716,
    '814649': 5.233256351039261,
    '901442': 5.745958429561201,
    '128935': 5.076212471131639,
    '214524': 5.196304849884527,
    '100610': 5.016166281755196,
    '134829': 5.3325635103926095,
    '116726': 5.914549653579677,
    '118225': 6.318706697459584,
    '134627': 6.348729792147806,
    '690152': 5.51270207852194,
    '102311': 5.741339491916859,
    '601127': 5.3325635103926095
}


'''#PSL Rest Dictionary
acw_values = {
    '654552': 5.781316348195329,
    '572045': 5.460743801652892,
    '789373': 7.326226012793177,
    '111514': 6.111111111111111,
    '135124': 6.283549783549783,
    '617748': 5.578723404255319,
    '320826': 5.814285714285714,
    '644246': 5.976394849785407,
    '130114': 5.602173913043479,
    '573249': 5.358606557377049,
    '725751': 6.647186147186147,
    '467351': 5.473333333333334,
    '108323': 7.274944567627495,
    '833249': 6.408247422680413,
    '765864': 6.116630669546436,
    '971160': 5.872651356993737,
    '562345': 6.116883116883117,
    '943862': 6.3822222222222225,
    '146129': 7.3180873180873185,
    '115017': 5.559006211180124,
    '966975': 7.336322869955157,
    '115825': 6.225531914893617,
    '706040': 5.628378378378378,
    '878877': 5.818947368421052,
    '132118': 6.1911764705882355,
    '104416': 6.266802443991853,
    '871762': 6.2877551020408164,
    '114823': 5.959139784946236,
    '581450': 6.115555555555556,
    '105923': 6.197002141327623,
    '872764': 7.254988913525499,
    '751550': 7.193684210526316,
    '145834': 5.563318777292577,
    '771354': 6.619658119658119,
    '140117': 5.877118644067797,
    '131722': 5.5073995771670194,
    '109123': 6.609375,
    '782561': 5.474332648870637,
    '818859': 5.880085653104925,
    '724446': 6.055679287305122,
    '770352': 6.200992555831266,
    '878776': 6.850102669404517,
    '263436': 5.8232848232848236,
    '126426': 6.194860813704497,
    '995174': 6.211065573770492,
    '385046': 6.907865168539326,
    '814649': 5.902222222222222,
    '901442': 6.539784946236559,
    '128935': 5.836206896551724,
    '214524': 5.948660714285714,
    '100610': 5.540481400437637,
    '134829': 6.684444444444445,
    '116726': 6.202272727272727,
    '118225': 6.965092402464066,
    '134627': 7.317865429234339,
    '690152': 6.415966386554622,
    '102311': 6.304347826086956,
    '601127': 6.370044052863436
}'''

# Convert the dictionary values to a list and calculate the mean ACW
acw_values_list = list(acw_values.values())
mean_acw = np.mean(acw_values_list)



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

# Function to detrend time series
# Function to detrend time series
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
    if time_series is cosine_similarity_ts:
        detrended_ts = time_series
        filtered_ts = detrended_ts
    else:
        detrended_ts = detrend_data(time_series)
        filtered_ts = apply_bandpass(detrended_ts)

    acw_series = []
    for i in range(0, len(filtered_ts) - window_size + 1, step_size):
        window = filtered_ts[i:i+window_size]
        acw_0 = calculate_acw(window)
        acw_series.append(acw_0)
    return acw_series


def calculate_te(source, target, m):
    # Ensure source and target have the same length
    min_length = min(len(source), len(target))
    source = np.array(source)[:min_length]  # Trim or pad source to match target's length
    target = np.array(target)[:min_length]  # Trim or pad target to match source's length

    # Create an array of tau values ranging from 1 to 10
    tau_values = np.arange(1, 11)  # Creates an array [1, 2, 3, ..., 10]

    # Initialize a list to store Transfer Entropy values
    te_results = []

    # Calculate Transfer Entropy for each tau value and store the results
    for tau in tau_values:
        te = te_compute.te_compute(source, target, m, tau)
        te_results.append(te)

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


def process_group(group_subjects, A1_tpn_df, acw_values, cosine_ts, iterations, m):
    
    cosine_acw_ts=get_acw_time_series(cosine_ts,150,1)
    mbb_s = [markov_block_bootstrap(cosine_similarity_ts, 66) for _ in range(1)]
    phi_s = [philipp_shuffle(cosine_similarity_ts, 66) for _ in range(1)]
    ran_s = [np.random.permutation(cosine_similarity_ts) for _ in range(1)]
    # Calculate TE without shuffling
    no_shuffle_acw_time_series = []
    no_shuffle_BOLD = []
    for subject_id in group_subjects:
        subject_ts = A1_tpn_df[str(subject_id)].values
        no_shuffle_BOLD.append(subject_ts)
        acw_ts = get_acw_time_series(subject_ts, 150, 1)
        no_shuffle_acw_time_series.append(acw_ts)
    no_shuffle_BOLD_mean=np.mean(no_shuffle_BOLD,axis=0)
    no_shuffle_BOLD_mean=no_shuffle_BOLD_mean[:len(cosine_ts)]
    no_shuffle_avg_acw_ts = np.mean(no_shuffle_acw_time_series, axis=0)
    no_shuffle_avg_acw_ts = no_shuffle_avg_acw_ts[:len(cosine_acw_ts)]
    true_te_forward = calculate_te(cosine_acw_ts, no_shuffle_avg_acw_ts, m)
    true_te_reverse = calculate_te(no_shuffle_avg_acw_ts, cosine_acw_ts,m )
    fals_te_forward = calculate_te(cosine_ts, no_shuffle_BOLD_mean, m)
    fals_te_reverse = calculate_te(no_shuffle_BOLD_mean,cosine_ts,m)

    # Calculate TE with shuffling for 'iterations' times
    for _ in range(iterations):

        mar_acw_time_series = []
        phi_acw_time_series = []
        ran_acw_time_series = []
        mar_time_series =[]
        phi_time_series =[]
        ran_time_series =[]
        
        for subject_id in group_subjects:
            subject_ts = A1_tpn_df[str(subject_id)].values

            block_length = int(np.ceil(acw_values[str(subject_id)]))
            mar_ts = markov_block_bootstrap(subject_ts, block_length)
            mar_time_series.append(mar_ts)
            phi_ts = philipp_shuffle(subject_ts, block_length)
            phi_time_series.append(phi_ts)
            ran_ts = np.random.permutation(subject_ts)
            ran_time_series.append(ran_ts)
            acw_ts_mar = get_acw_time_series(mar_ts, 150, 1)
            mar_acw_time_series.append(acw_ts_mar)
            acw_ts_phi = get_acw_time_series(phi_ts, 150, 1)
            phi_acw_time_series.append(acw_ts_phi)
            acw_ts_ran = get_acw_time_series(ran_ts,150,1)
            ran_acw_time_series.append(acw_ts_ran)
        
        mar_avg_acw_ts = np.mean(mar_acw_time_series, axis=0)
        mar_avg_acw_ts = mar_avg_acw_ts[:len(cosine_acw_ts)]
        phi_avg_acw_ts = np.mean(phi_acw_time_series, axis=0)
        phi_avg_acw_ts = phi_avg_acw_ts[:len(cosine_acw_ts)]
        ran_avg_acw_ts = np.mean(ran_acw_time_series, axis=0)
        ran_avg_acw_ts = ran_avg_acw_ts[:len(cosine_acw_ts)]


    # Calculate p-values for ACW TE
    p_values_acw_forward_mbb = p_value(cosine_acw_ts, mar_acw_time_series, true_te_forward)
    p_values_acw_reverse_mbb = p_value(no_shuffle_avg_acw_ts, [get_acw_time_series(ts) for ts in mbb_s], true_te_reverse)
    p_values_acw_forward_phi = p_value(cosine_acw_ts, phi_acw_time_series, true_te_forward)
    p_values_acw_reverse_phi = p_value(no_shuffle_avg_acw_ts, [get_acw_time_series(ts) for ts in phi_s], true_te_reverse)
    p_values_acw_forward_ran = p_value(cosine_acw_ts, ran_acw_time_series, true_te_forward)
    p_values_acw_reverse_ran = p_value(no_shuffle_avg_acw_ts, [get_acw_time_series(ts) for ts in ran_s], true_te_reverse)

    # Calculate p-values for raw TE
    p_values_raw_forward_mbb = p_value(cosine_similarity_ts, mar_time_series, fals_te_forward)
    p_values_raw_forward_phi = p_value(cosine_similarity_ts, phi_time_series, fals_te_forward)
    p_values_raw_forward_ran = p_value(cosine_similarity_ts, ran_time_series, fals_te_forward)
    p_values_raw_reverse_mbb = p_value(no_shuffle_BOLD_mean, mbb_s, fals_te_reverse)
    p_values_raw_reverse_phi = p_value(no_shuffle_BOLD_mean, phi_s, fals_te_reverse)
    p_values_raw_reverse_ran = p_value(no_shuffle_BOLD_mean, ran_s, fals_te_reverse)
    

    return true_te_forward,true_te_reverse,fals_te_forward,fals_te_reverse, p_values_acw_forward_mbb,p_values_acw_forward_phi,p_values_acw_forward_ran,p_values_acw_reverse_mbb,p_values_acw_reverse_phi,p_values_acw_reverse_ran,p_values_raw_forward_mbb,p_values_raw_forward_phi,p_values_raw_forward_ran,p_values_raw_reverse_mbb,p_values_raw_reverse_phi,p_values_raw_reverse_ran


'''def process_group(group_subjects, A1_tpn_df, acw_values, cosine_ts, iterations, m):
    no_shuffle_te_values = []
    shuffle_te_values = {tau: [] for tau in range(1, 11)}
    p_values = {tau: 0 for tau in range(1, 11)}

    # Calculate TE without shuffling using raw time series
    no_shuffle_time_series = []
    for subject_id in group_subjects:
        subject_ts = A1_tpn_df[str(subject_id)].values
        no_shuffle_time_series.append(subject_ts)
    no_shuffle_avg_ts = np.mean(no_shuffle_time_series, axis=0)
    no_shuffle_avg_ts = no_shuffle_avg_ts[:len(cosine_ts)]
    no_shuffle_te_values = calculate_te(cosine_ts, no_shuffle_avg_ts, m)

    # Calculate TE with shuffling for 'iterations' times using raw time series
    # Calculate TE with shuffling for 'iterations' times using raw time series
    for _ in range(iterations):
        shuffle_time_series = []
        for subject_id in group_subjects:
            subject_ts = A1_tpn_df[str(subject_id)].values.copy()  # Copy the array to avoid modifying the original

            # Apply random shuffling
            np.random.shuffle(subject_ts)
            shuffle_time_series.append(subject_ts)

        shuffle_avg_ts = np.mean(shuffle_time_series, axis=0)
        shuffle_avg_ts = shuffle_avg_ts[:len(cosine_ts)]

        shuffle_te = calculate_te(cosine_ts, shuffle_avg_ts, m)
        for tau in range(1, 11):
            shuffle_te_values[tau].append(shuffle_te[tau - 1])


    # Calculate p-values
    for tau in range(1, 11):
        no_shuffle_te = no_shuffle_te_values[tau - 1]
        greater_count = sum(te > no_shuffle_te for te in shuffle_te_values[tau])
        p_values[tau] = greater_count / iterations

    return no_shuffle_te_values, shuffle_te_values, p_values'''



def print_acw_values(subject_id, A1_tpn_df, acw_values):
    subject_ts = A1_tpn_df[str(subject_id)].values
    block_length = int(np.ceil(acw_values[str(subject_id)])) + 3
    acw_ts = get_acw_time_series(subject_ts, 150, 1)
    print(f"ACW Time Series for Subject {subject_id}: First 10 values - {acw_ts[:10]}, Last 10 values - {acw_ts[-10:]}")

def process_all_subjects(A1_tpn_df, acw_values, cosine_ts, iterations, m):
    all_subjects = list(acw_values.keys())
    return process_group(all_subjects, A1_tpn_df, acw_values, cosine_ts, iterations, m)





# Calculate the ACW time series for the cosine similarity without preprocessing
cosine_acw_ts = get_acw_time_series(cosine_ts, 150, 1)


print(f"Cosine Similarity Time Series: First 10 values - {cosine_ts[:10]}, Last 10 values - {cosine_ts[-10:]}")

cosine_acw_ts = get_acw_time_series(cosine_ts, 150, 1)
print(f"Cosine Similarity ACW Time Series: First 10 values - {cosine_acw_ts[:10]}, Last 10 values - {cosine_acw_ts[-10:]}")

# Print ACW values for specific subjects and all subjects
print_acw_values('654552', A1_tpn_df, acw_values)
print_acw_values('111514', A1_tpn_df, acw_values)
print_acw_values('601127', A1_tpn_df, acw_values)

# Group subjects based on their ACW
high_acw_subjects = [s for s in acw_values if acw_values[s] > mean_acw]
low_acw_subjects = [s for s in acw_values if acw_values[s] <= mean_acw]
# Parameters

iterations=1
m = 3
taus = range(1, 11)

# Process all subjects and print ACW values for the average of all subjects
all_subjects_te_values = process_all_subjects(A1_tpn_df, acw_values, cosine_ts, iterations, m)
avg_acw_ts = np.mean([get_acw_time_series(A1_tpn_df[str(subject_id)].values, 150, 1) for subject_id in acw_values.keys()], axis=0)
print(f"Average ACW Time Series of All Subjects: First 10 values - {avg_acw_ts[:10]}, Last 10 values - {avg_acw_ts[-10:]}")


# Process high, low, and all subjects groups
high_acw_results = process_group(high_acw_subjects, A1_tpn_df, acw_values, cosine_ts, iterations, m)
low_acw_results = process_group(low_acw_subjects, A1_tpn_df, acw_values, cosine_ts, iterations, m)
all_subjects_results = process_all_subjects(A1_tpn_df, acw_values, cosine_ts, iterations, m)

# Inside the loop for CSV saving
for group_name, (true_te_forward,true_te_reverse,fals_te_forward,fals_te_reverse, p_values_acw_forward_mbb,p_values_acw_forward_phi,p_values_acw_forward_ran,p_values_acw_reverse_mbb,p_values_acw_reverse_phi,p_values_acw_reverse_ran,p_values_raw_forward_mbb,p_values_raw_forward_phi,p_values_raw_forward_ran,p_values_raw_reverse_mbb,p_values_raw_reverse_phi,p_values_raw_reverse_ran) in [('L', low_acw_results), ('H', high_acw_results), ('All', all_subjects_results)]:
    results = []
    
    results.append({
        'te_acw_forward': true_te_forward,
        'te_acw_reverse': true_te_reverse,
        'te_raw_forward': fals_te_forward,
        'te_raw_reverse': fals_te_reverse,
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
    })
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/Users/satrokommos/Desktop/Euclidean/TEWDnew/A1_WDoldcode1609Aug12_{group_name}.csv', index=False)
