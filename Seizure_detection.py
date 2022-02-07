import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq , rfft, rfftfreq
from sklearn.model_selection import KFold
# import pyeeg 
# from eeglib import features
import pywt #py wavelets pywt.wavedec()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif,SelectKBest, SelectPercentile #for mutual info
from sklearn.decomposition import PCA
from scipy.stats import ranksums # for wilcoxon function
from scipy import signal as sig
from scipy.stats import skew,kurtosis
from sklearn.model_selection import cross_validate
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

def split_data(X,Y,split_ratio = 0.2,normalization=True):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=1)
    if normalization:
        Scaler_tr = StandardScaler().fit(X_train)
        X_train = Scaler_tr.transform(X_train)
        X_test = Scaler_tr.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test
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


def peak_frequency(signal):
    peak_freq = []
    ft = fft(signal)
    mag = np.abs(ft)
    freq = fftfreq(mag.size,1/256)
    
    cutoffs = [0.5,4,8,12,25]
    for i in range(len(cutoffs)-1):
        cut_freq = freq[np.logical_and(freq>cutoffs[i],freq<cutoffs[i+1])]
        cut_mag = mag[np.logical_and(freq>cutoffs[i],freq<cutoffs[i+1])]
        peak_freq.append(cut_freq[cut_mag==max(cut_mag)][0])

    return peak_freq

def extract_features(signal):
    peak_freq = []
    rms = []
    entropy = []
    stat_features = []
    for trial in range(signal.shape[0]):
        trial_stat = []
        trial_freq = []
        trial_rms = []
        trial_entropy = []
        
        for electrode in range(signal.shape[2]):
            trial_rms.append(np.sqrt(np.mean(signal[trial,:,electrode]**2)))
            trial_entropy.append(np.sum(signal[trial,:,electrode]*np.log(signal[trial,:,electrode]**2)))
            trial_freq += peak_frequency(signal[trial,:,electrode])
            trial_stat.append(np.mean(signal[trial,:,electrode]))
            trial_stat.append(np.var(signal[trial,:,electrode]))
            trial_stat.append(skew(signal[trial,:,electrode]))
            trial_stat.append(kurtosis(signal[trial,:,electrode]))

            coeff = pywt.wavedec(signal[trial,:,electrode],'db4',level=5)
            for level in range(len(coeff)):
                trial_stat.append(np.mean(coeff[level],axis =0))
                trial_stat.append(np.var(coeff[level],axis =0))
                trial_stat.append(skew(coeff[level],axis =0))
                trial_stat.append(kurtosis(coeff[level],axis =0))
        peak_freq.append(trial_freq)
        rms.append(trial_rms)
        entropy.append(trial_entropy)
        stat_features.append(trial_stat)
    
    features = np.concatenate((peak_freq,rms,entropy,stat_features),axis=1)
    return features


def pypackage_features():
    # use popular python packages to extract features from eeg signals (pyeeg,EEGlib,Cesuim)
    return package_features

# Feature selection & reduction


def mutual_info(features, cdf):
    # use mutual information as a criteria to select the best features, and return them
    select_percentile = SelectPercentile(
        mutual_info_classif, cdf)  # Here cdf should be 10 for 10%
    # Not sure if this the way to do it, but here we go
    selected_features = select_percentile.fit_transform(features)
    return selected_features


def wilcoxon_test(X_train,X_test,X_cv, Y_train,threshold):
    # use wilcoxon test as a criteria to select the best features, and return them
    selected_features = []
    
    class_1 = X_train[np.where(Y_train==0)]
    class_2 = X_train[np.where(Y_train==1)]
    p_Values = []
    for i in range(X_train.shape[1]):
        feature_class_1 = class_1[:,i]
        feature_class_2 = class_2[:,i]
        _,p_Value = ranksums(feature_class_1,feature_class_2)
        p_Values.append(p_Value)
    p_Values = np.asarray(p_Values)
    X_train_reduced = np.squeeze(X_train[:,np.where(p_Values<threshold)])
    X_test_reduced = np.squeeze(X_test[:,np.where(p_Values<threshold)])
    X_cv_reduced = np.squeeze(X_cv[:,np.where(p_Values<threshold)])
    return X_train_reduced,X_test_reduced,X_cv_reduced

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

X_train, X_test, Y_train, Y_test = split_data(extract_features(all_trials),all_labels, 0.4)
trainer = SVC(kernel='sigmoid')
trainer.fit(X_train,Y_train)
print(accuracy_score(Y_test,trainer.predict(X_test)))
# -------------------------
X_test, X_cv, y_test, y_cv = split_data(X_test, Y_test, 0.5, False)

# X_train_reduced, X_test_reduced, X_cv_reduced = mutual_info(X_train, X_test, X_cv, Y_train, 10)
X_train_reduced, X_test_reduced, X_cv_reduced = wilcoxon_test(X_train, X_test, X_cv, Y_train,0.005)
# X_train_reduced, X_test_reduced, X_cv_reduced = pca(X_train, X_test, X_cv, 20)

trainer = SVC(kernel='sigmoid')
trainer.fit(X_train_reduced,Y_train)
print(accuracy_score(y_cv,trainer.predict(X_cv_reduced)), end=' ')
# for train_Idx, test_Idx in kfold.split(all_labels):
# call preprocessing
# call feature extraction
# call feature selection
# call SVM classifier and output the result
