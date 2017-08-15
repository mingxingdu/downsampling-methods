# !/usr/bin/env python
#
# Author : the Zero Resource Challenge Team
#
# In this Module, we take as input features and a
# desired number of output samples, and we output
# a downsampled version of the features.

import os
import sys
import numpy as np

def downsample(feats, n_samples, parameter1, parameter2):
    ''' downsample

    returns n_samples features sampled at a non-uniform interval
    with a triangular weigthing scheme 

    Parameters
    ----------
    feats : numpy.ndarray
       Array with continous features (MFCC, PLP, etc)

    n_samples: int
       the number of features to return
    
    parameter1: float
        non-applicable for this method

    parameter2: float
        the ratio of the max interval to the min interval of the downsampling non-uniform
       

    Returns
    -------
    downsampled_feats: numpy.ndarray
        the downsampled features

    downsampled_indexes: numpy.ndarray
        lower and upper indexes of


    '''
    alpha = parameter2
    mid = (n_samples-1)/2
    n_feats = feats.shape[0]
    indexes = np.arange(n_feats)
    # sampling non-uniform
    # according to the ratio max/min, calculate the indexes of the dowmsampled features.
    if n_samples % 2 == 0 :
        min_dis = (n_feats-1.0)/((alpha-1)*mid + 2*mid +1)
        skewness = (alpha-1.0)*min_dis / mid
        left = skewness*np.arange(mid).cumsum() + min_dis*(np.arange(mid) + 1)
        right = (n_feats-1 - left)[::-1]
        samples = np.append(left, right)
    else : 
        min_dis = (n_feats-1.0)/((alpha-1)*mid + 2*mid)
        skewness = (alpha-1.0)*min_dis / (mid-1)
        left = skewness*np.arange(mid).cumsum() + min_dis*(np.arange(mid) + 1)
        right = (n_feats-1 - left)[:-1][::-1]
        samples = np.append(left, right)

    samples = np.append(0, samples)
    samples = np.append(samples, n_feats-1)

    indexes_floor = np.floor(samples).astype(int)
    indexes_ceil = np.ceil(samples).astype(int)
    proportion = (samples - np.trunc(samples))[:, None]

    downsampled_feats = (feats[indexes_floor] * (1 - proportion) +
                         feats[indexes_ceil] * proportion)
    return downsampled_feats
