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

    returns n_samples features sampled at a non-uniform interval
    with a regular gaussian weigthing scheme

    Parameters
    ----------
    feats : numpy.ndarray
       Array with continous features (MFCC, PLP, etc)

    n_samples: int
       the number of features to return
    
    parameter1: float
        the standard deviation of the gaussian filter

    parameter2: float
        the ratio of the max interval to the min interval of the downsampling non-uniform

    Returns
    -------
    downsampled_feats: numpy.ndarray
        the downsampled features

    downsampled_indexes: numpy.ndarray
        lower and upper indexes of


    '''   
    # alpha = max/min & n_samples is pair

    delta = parameter1
    alpha = parameter2

    mid = (n_samples-1)/2
    n_feats = feats.shape[0]
    indexes = np.arange(n_feats+1)
    
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
    samples = samples.reshape(n_samples, 1)
    # initialize the downsampled features as 0
    downsampled_feats = np.zeros((n_samples, feats.shape[1]))
    # generate and apply faussian filters to the input, by using matrix operation`
    ''' gaussian filter is of format n_samples * n_feats
        each row corresponds to a gaussian filter for one downsampling output 
    '''
    indexes = np.repeat(indexes.reshape(1, n_feats+1), n_samples, axis = 0)
    indexes = indexes - samples
    integral = norm.cdf(indexes, scale = delta)
    intervals = np.array([integral[:,i+1]-integral[:,i] for i in range(n_feats)]).T
    # normolize the gaussian filter
    gaussian_filter = intervals / intervals.sum(axis = 1).reshape(n_samples, 1)
    downsampled_feats = np.dot(gaussian_filter, feats)

    return downsampled_feats
