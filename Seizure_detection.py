import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pyeeg 
from eeglib import features
import pywt #py wavelets pywt.wavedec()
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif,SelectKBest, SelectPercentile #for mutual info
from sklearn.decomposition import PCA
from scipy.stats import ranksums # for wilcoxon function

eeg_sampling =256
#Preprocessing
def apply_filter(signal, a=None,b=None):
    # apply a bandpass filter to the signal where a and b are the corner frequencies
    
    return filtered_signal
def filter_bank():

    return filtered_signals
def normalization(features):
    # mean and min-max normalization
    return norm_features
def subset_trials(signal,trial_length):
    # subset the loaded signal to trials of a certain length and return it as a np array
    signal = np.asarray(signal)
    samples_per_trial = (trial_length*60*eeg_sampling)
    num_trials = int(np.floor(signal.shape[0]/samples_per_trial))
    subsetted_signal = np.zeros([num_trials,samples_per_trial,23])
   
    for trial in range(num_trials):
        for electrode in range(23):
            trial_start = samples_per_trial*trial
            trial_end = trial_start + samples_per_trial
            subsetted_signal[trial,:,electrode] = signal[trial_start:trial_end,electrode]

    if (signal[0,-1]==0):
        labels = np.zeros(num_trials)
    else:
        labels = np.ones(num_trials)
    return subsetted_signal,labels # trials * samples * electrodes


    #Feature extraction
def freq_features():
    #Compute frequency features for signal (from paper)
    return frequency_features
def stat_features():
    # compute mean skewness variance and kurtosis on time domain signal and wavelet coefficients
    return statistical_features
def pypackage_features():
    #use popular python packages to extract features from eeg signals (pyeeg,EEGlib,Cesuim)
    return package_features

#Feature selection & reduction
def mutual_info(features,cdf):
    # use mutual information as a criteria to select the best features, and return them
    return selected_features
def wilcoxon_test(features,p_value):
    # use wilcoxon test as a criteria to select the best features, and return them
     return selected_features
def pca(features,variance):
    # use PCA for feature reduction with a specified variance to keep.
    return reduced_features


#Load signal 
data = pd.read_csv('chbmit_preprocessed_data.csv')
data.columns
grouped_data = data.groupby('Outcome')
preictal_data = grouped_data.get_group(0)
ictal_data = grouped_data.get_group(1)
#  data exploration 
print(preictal_data.columns)
print(preictal_data.shape)
print(ictal_data.shape)
print(preictal_data.shape[0]/(256*60)) # print length of recording in minutes, 256 is sampling frequency

# create trials and split data to trian and test
subsetted_preictal, labels_preictal = subset_trials(preictal_data,1)
subsetted_ictal, labels_ictal = subset_trials(ictal_data,1)
all_trials = np.concatenate((subsetted_preictal,subsetted_ictal), axis=0)
all_labels = np.concatenate((labels_preictal,labels_ictal), axis=0) #Zero is preictal, 1 is ictal


kfold = KFold(n_splits= 10,shuffle= True, random_state=42)
# for train_Idx, test_Idx in kfold.split(all_labels):
    # call preprocessing
    # call feature extraction
    # call feature selection
    # call SVM classifier and output the result