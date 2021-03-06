# Seizure-Detection-CHB-MIT-dataset
in this project we try to build a ML model that can classify seizure occurance in humans using EEG recordings. the dataset used is the CHB-MIT dataset.
---
## Dataset
> The CHB-MIT dataset consists of EEG recordings 24 participants, with 23 electrodes. the final column is the outcome column, with 0 indicating preictal, and 1 indicating ictal. 
---
## Preprocessing 
for preprocessing, the following steps are implemented:
1. Filtering: as per literature, the functional frequency range for seizure detection is 2-20 Hz. However, most literature suggest using a pandbass filter with corner frequencies of 0.5 Hz and 36 Hz.
2. Filter Banks: after the initial filter, we may apply filters to study each frequency band of interest on its own, namely, the delta, theta, alpha, beta and gamma bands. 
3. Standardize features by removing the mean and scaling to unit variance. (these steps will be done after feature extraction)
---
## Feature Extraction
for this step, we are going to implement some useful features that were used in literature, as well as some features from renown EEG python libraries. some examples:
1. frequency domain features: power spectral density, peak frequency, median frequency. 
2. statistical features on time domain signal: mean, variance, skewness and kurtosis.
3. python packages: pyeeg.
---
## Dimensionality reduction and feature selection
further assessment of the dimensionality of the extracted features is needed before we conclude a plan for this section of the model. However, we are using Princible component analysis for dimensionality reduction, and combining it with a filter method for selection.
### candidate filter methods: 
1- mutual information  
2- univariate statistical test (such as wilcoxon or t-test) 
---
## Classification
Support vector machines (SVM) is one of the most widely used classifiers in literature, and is also computationally efficient, hence we choose it for our classification problem.
---

## Install dependencies
```
$ pip install -r requirements.txt
```
#### To install pyeeg
```
$ git clone https://github.com/forrestbao/pyeeg.git
$ cd pyeeg
$ python setup.py install
``` 
---
## Dataset
The dataset can be downloaded by clicking [Here](https://ieee-dataport.s3.amazonaws.com/open/65970/chbmit_preprocessed_data.csv?response-content-disposition=attachment%3B%20filename%3D%22chbmit_preprocessed_data.csv%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20220208%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220208T202215Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=3600&X-Amz-Signature=3a302a134e369844f47e2e2980324db247e0c590717f5f7226d2cd6191ad9771) 
---


# Team
1. Ammar Al-Saeed Mohammed Section: 2, BN: 1.
2. Ahmed Sayed Elbadawy Section: 1, BN: 4.
3. Ramadan Ibrahim Section: 1, BN: 34.
