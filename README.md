# Seizure-Detection-CHB-MIT-dataset
in this project we try to build a ML model that can classify seizure occurance in humans using EEG recordings. the dataset used is the CHB-MIT dataset.
---
## Dataset
> The CHB-MIT dataset consists of EEG recordings 24 participants, with 23 electrodes. the final column is the outcome column, with 0 indicating preictal, and 1 indicating ictal. 
---
## Preprocessing 
for preprocessing, the following steps are needed:
1. Filtering: as per literature, the functional frequency range for seizure detection is 2-20 Hz. However, most literature suggest using a pandbass filter with corner frequencies of 0.5 Hz and 36 Hz.
2. Temporal analysis: we may try to devide our data to incremental windows primarily to determine how long data acquisition it would take for the model to provide decent accuracy.
3. Filter Banks: after the initial filter, we may apply filters to study each frequency band of interest on its own, namely, the delta, theta, alpha, beta and gamma bands. 
4. Mean normalization, min-max normalization. (these steps will be done after feature extraction)
---
## Feature Etraction
for this step, we are going to implement some useful features that were used in literature, as well as some features from renown EEG python libraries. some examples:
1. frequency domain features: powers pectral density, peak frequency, median frequency, spectral entropy.
2. wavelet features: mean, variance, skewness and kurtosis of wavelet coefficients. 
3. statistical features on time domain signal: the same 4 mentioned above.
4. python packages: pyeeg, Gumpy and EEGlib. 
---
## Dimensionality reduction and feature selection
further assessment of the dimensionality of the extracted features is needed before we conclude a plan for this section of the model. However, we may consider using Princible component analysis for dimensionality reduction, and combining it with a filter method or a wrapper method for selection depending on computational resources and complexity.
### candid filter methods: 
1- mutual information  
2- univariate statistical test (such as wilcoxon or t-test) 
### candid wrapper methods:
1- Sequential forward search
2- sequential backward search
3- bidirectional search

---
## Classification
Support vector machines (SVM) is one of the most widely used classifiers in literature, and is also computationally efficient, hence we choose it for our classification problem.
