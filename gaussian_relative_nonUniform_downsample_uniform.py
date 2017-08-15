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
from scipy.stats import norm



def downsample(feats, n_samples, parameter1, parameter2):
    ''' downsample

    returns n_samples features sampled at a regular interval
    with a gaussian filter, its standard deviation is variant according to the length of the input and also the position of the downsampling.

    Parameters
    ----------
    feats : numpy.ndarray
       Array with continous features (MFCC, PLP, etc)

    n_samples: int
       the number of features to return
    
    parameter1: float
        the ratio of the standard deviation with respect to the length of the input feats.

    parameter2: float 
        the slope for the linear transformation of the standard deviation. 

    Returns
    -------
    downsampled_feats: numpy.ndarray
        the downsampled features

    downsampled_indexes: numpy.ndarray
        lower and upper indexes of


    '''    
    
    coefficient = parameter1
    slope = parameter2


    n_feats = feats.shape[0]
    indexes = np.arange(n_feats+1)
    # downsampling uniform
    samples = np.linspace(0, n_feats-1, num = n_samples).reshape(n_samples, 1)
    
    downsampled_feats = np.zeros((n_samples, feats.shape[1]))
    # linear transformation for the standard deviation, according to the downsampling position.  
    if n_samples % 2 ==0 :
        scales = np.arange(n_samples/2)
        scales = np.append(scales, scales[::-1])
        scales = scales * slope + coefficient * n_feats
    else :
        scales = np.arange(n_samples/2 + 1)
        scales = np.append(scales, scales[:-1][::-1])
        scales = scales * slope + coefficient * n_feats
    scales = scales.reshape(n_samples, 1)
    # generate and apply faussian filters to the input, by using matrix operation`  
    ''' gaussian filter is of format n_samples * n_feats
        each row corresponds to a gaussian filter for one downsampling output 
    '''
    indexes = np.repeat(indexes.reshape(1, n_feats+1), n_samples, axis = 0)
    indexes = indexes - samples
    integral = norm.cdf(indexes, scale = scales)
    intervals = np.array([integral[:,i+1]-integral[:,i] for i in range(n_feats)]).T
    # normalize the gaussian filter
    gaussian_filter = intervals / intervals.sum(axis = 1).reshape(n_samples, 1)
    downsampled_feats = np.dot(gaussian_filter, feats)

    return downsampled_feats
