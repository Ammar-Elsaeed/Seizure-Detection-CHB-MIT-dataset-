import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pyeeg
from eeglib import features
import pywt  # py wavelets pywt.wavedec()
from sklearn import svm
# for mutual info
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectPercentile
from sklearn.decomposition import PCA
from scipy.stats import ranksums  # for wilcoxon function
from scipy import signal as sig
from scipy.stats import skew, kurtosis
eeg_sampling = 256
# Preprocessing


def apply_filter(signal, low_freq=None, high_freq=None):
    # apply a bandpass filter to the signal where a and b are the corner frequencies
    low_freq = low_freq/eeg_sampling
    high_freq = high_freq/eeg_sampling
    b, a = sig.butter(1, [low_freq, high_freq], btype="bandpass")
    filtered_signal = sig.filtfilt(b, a, signal, axis=1)
    return filtered_signal


def filter_bank(signal, cutoffs=None):
    if (cutoffs == None):
        cutoffs = [0.5, 4, 8, 12, 25]
    filtered_signals = np.empty(
        [len(cutoffs)-1, signal.shape[0], signal.shape[1], signal.shape[2]])  # 4 * 68 * 15360 * 23
    for band in range(len(cutoffs)-1):
        bands = [cutoffs[band], cutoffs[band+1]]
        filtered_signals[band, :, :, :] = apply_filter(
            signal, low_freq=bands[0], high_freq=bands[1])
    return filtered_signals


def normalization(train_features, test_features=None):
    # mean and min-max normalization
    norm_train_features = np.empty(
        [train_features.shape[0], train_features.shape[1]])
    if (test_features == None):
        for feature in range(train_features.shape[1]):
            train_column = train_features[:, feature]
            norm_train_features[:, feature] = (
                train_column - np.min(train_column)) / (np.max(train_column)-np.min(train_column))
        return norm_train_features
    else:
        norm_test_features = np.empty(
            [test_features.shape[0], test_features.shape[1]])
        for feature in range(train_features.shape[1]):
            train_column = train_features[:, feature]
            norm_train_features[:, feature] = (
                train_column - np.min(train_column)) / (np.max(train_column)-np.min(train_column))
            test_column = test_features[:, feature]
            norm_test_features[:, feature] = (
                test_column - np.min(train_column)) / (np.max(train_column)-np.min(train_column))
        return norm_train_features, norm_test_features


def subset_trials(signal, trial_length):
    # subset the loaded signal to trials of a certain length and return it as a np array
    signal = np.asarray(signal)
    samples_per_trial = (trial_length*60*eeg_sampling)
    num_trials = int(np.floor(signal.shape[0]/samples_per_trial))
    subsetted_signal = np.zeros([num_trials, samples_per_trial, 23])

    for trial in range(num_trials):
        for electrode in range(23):
            trial_start = samples_per_trial*trial
            trial_end = trial_start + samples_per_trial
            subsetted_signal[trial, :,
                             electrode] = signal[trial_start:trial_end, electrode]

    if (signal[0, -1] == 0):
        labels = np.zeros(num_trials)
    else:
        labels = np.ones(num_trials)
    return subsetted_signal, labels  # trials * samples * electrodes

    # Feature extraction


def freq_features():
    # Compute frequency features for signal (from paper)
    return frequency_features


def stat_features(signal):
    # compute mean skewness variance and kurtosis on time domain signal and wavelet coefficients
    statistical_features = []
    for trial in range(signal.shape[0]):
        electrodes_features = []
        for electrode in range(signal.shape[2]):
            electrodes_features.append(np.mean(signal[trial, :, electrode]))
            electrodes_features.append(np.var(signal[trial, :, electrode]))
            electrodes_features.append(skew(signal[trial, :, electrode]))
            electrodes_features.append(kurtosis(signal[trial, :, electrode]))

            coeff = pywt.wavedec(signal[trial, :, electrode], 'db4', level=5)
            for level in range(len(coeff)):
                electrodes_features.append(np.mean(coeff[level], axis=0))
                electrodes_features.append(np.var(coeff[level], axis=0))
                electrodes_features.append(skew(coeff[level], axis=0))
                electrodes_features.append(kurtosis(coeff[level], axis=0))

        statistical_features.append(electrodes_features)
    statistical_features = np.asarray(statistical_features)
    # 136 * 644. 644: 23 electrode * 28 feature. 28 feature: 4 time, 6*4 wavelets
    return statistical_features


def pypackage_features():
    # use popular python packages to extract features from eeg signals (pyeeg,EEGlib,Cesuim)
    return package_features

# Feature selection & reduction


def mutual_info(features, cdf):
    # use mutual information as a criteria to select the best features, and return them
    return selected_features


def wilcoxon_test(features, p_value):
    # use wilcoxon test as a criteria to select the best features, and return them
    return selected_features


def pca(features, variance):
    # use PCA for feature reduction with a specified variance to keep.
    pca = PCA(n_components=variance, svd_solver='full')
    principal_components = pca.fit_transform(features)
    reduced_features = pd.DataFrame(data=principal_components)
    return reduced_features


# Load signal
data = pd.read_csv('chbmit_preprocessed_data.csv')
data.columns
grouped_data = data.groupby('Outcome')
preictal_data = grouped_data.get_group(0)
ictal_data = grouped_data.get_group(1)
#  data exploration
print(preictal_data.columns)
print(preictal_data.shape)
print(ictal_data.shape)
# print length of recording in minutes, 256 is sampling frequency
print(preictal_data.shape[0]/(256*60))

# create trials and split data to trian and test
subsetted_preictal, labels_preictal = subset_trials(preictal_data, 1)
subsetted_ictal, labels_ictal = subset_trials(ictal_data, 1)
all_trials = np.concatenate((subsetted_preictal, subsetted_ictal), axis=0)
# Zero is preictal, 1 is ictal
all_labels = np.concatenate((labels_preictal, labels_ictal), axis=0)
stat = stat_features(all_trials)
filtered_signals = filter_bank(all_trials)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# for train_Idx, test_Idx in kfold.split(all_labels):
# call preprocessing
# call feature extraction
# call feature selection
# call SVM classifier and output the result
