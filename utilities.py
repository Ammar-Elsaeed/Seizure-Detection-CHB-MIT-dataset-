import numpy as np
from numpy.fft import fft, fftfreq
import pyeeg 
import pywt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectPercentile 
from sklearn.decomposition import PCA
from scipy.stats import ranksums 
from scipy import signal as sig
from scipy.stats import skew,kurtosis

eeg_sampling =256
#Preprocessing
def apply_filter(signal, low_freq=None,high_freq=None):
    # apply a bandpass filter to the signal where a and b are the corner frequencies
    low_freq = low_freq/eeg_sampling
    high_freq = high_freq/eeg_sampling
    b,a = sig.butter(1,[low_freq,high_freq], btype="bandpass")
    filtered_signal = sig.filtfilt(b,a,signal,axis=1) 
    return filtered_signal
def filter_bank(signal,cutoffs=None):
    if (cutoffs == None):
        cutoffs = [0.5,4,8,12,25]
    filtered_signals = np.empty([len(cutoffs)-1,signal.shape[0],signal.shape[1],signal.shape[2]]) # 4 * 68 * 15360 * 23
    for band in range(len(cutoffs)-1):
        bands = [cutoffs[band],cutoffs[band+1]]
        filtered_signals[band,:,:,:] = apply_filter(signal, low_freq=bands[0],high_freq=bands[1])
    return filtered_signals
def split_data(X,Y,split_ratio = 0.2,normalization=True):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=1)
    if normalization:
        Scaler_tr = StandardScaler().fit(X_train)
        X_train = Scaler_tr.transform(X_train)
        X_test = Scaler_tr.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test
    
def subset_trials(signal,trial_length):
    # subset the loaded signal to trials of a certain length and return it as a np array
    signal = np.asarray(signal)
    samples_per_trial = (trial_length*60*eeg_sampling)
    num_trials = int(np.floor(signal.shape[0]/samples_per_trial))
    subsetted_signal = np.empty([num_trials,samples_per_trial,23])
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


def pypackage_features(signal,band=None):
#use popular python packages to extract features from eeg signals
    package_features = []
    if (band == None):
        band = [0.5,4,8,12,25]
    for trial in range(signal.shape[0]):
        trial_features = []
        for electrode in range(signal.shape[2]):
           _,power_ratio = pyeeg.bin_power(signal[trial,:,electrode],Band=band,Fs=eeg_sampling)
           trial_features.append(pyeeg.spectral_entropy(signal[trial,:,electrode],Band=band,Fs=eeg_sampling,Power_Ratio=power_ratio))
        package_features.append(trial_features)
    package_features = np.asarray(package_features)
    return package_features

#Feature selection & reduction
def mutual_info(X_train, X_test, X_cv, y_train, cdf):
    # use mutual information as a criteria to select the best features, and return them
    select_percentile = SelectPercentile(
        mutual_info_classif, percentile = cdf)  # Here cdf should be 10 for 10%
    # Not sure if this the way to do it, but here we go
    selected_features_train = select_percentile.fit_transform(X_train, y_train)
    selected_features_test = select_percentile.transform(X_test)
    selected_features_cv = select_percentile.transform(X_cv)
    return selected_features_train, selected_features_test, selected_features_cv
def wilcoxon_test(X_train,X_test,X_cv, Y_train,threshold):
    # use wilcoxon test as a criteria to select the best features, and return them
   
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
def pca(X_train, X_test, X_cv, variance):
    # use PCA for feature reduction with a specified variance to keep.
    pca = PCA(n_components=variance)
    principal_components_train = pca.fit_transform(X_train)
    principal_components_cv = pca.transform(X_cv)
    principal_components_test = pca.transform(X_test)
    return principal_components_train, principal_components_test, principal_components_cv