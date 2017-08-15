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

    returns n_samples features sampled at a regular interval, choosing the nearest neighbour as the downsampled feature.

    Parameters
    ----------
    feats : numpy.ndarray
       Array with continous features (MFCC, PLP, etc)

    n_samples: int
       the number of features to return
    
    parameter1: float
        non-applicable in this method
    
    parameter2: float
        non-applicable in this method
    
    Returns
    -------
    downsampled_feats: numpy.ndarray
        the downsampled features

    downsampled_indexes: numpy.ndarray
        lower and upper indexes of


    '''
    n_feats = feats.shape[0]
    samples = np.linspace(0, n_feats-1, num=n_samples)
    indexes = np.rint(samples).astype(int)

    downsampled_feats = feats[indexes]

    return downsampled_feats
