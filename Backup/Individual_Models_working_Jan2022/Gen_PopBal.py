# -*- coding: utf-8 -*-
"""
Created in 2021

@author: Robbie
"""
from scipy import interpolate
import numpy as np

#------------------------------------------------------------------------
### Create the cumalitive distributions from mass distributions       ###
#------------------------------------------------------------------------
def get_cumdist(data):
    # This function assumes the 1st entry in the array corresponds to the biggest sieve
    # data - a 1D array containing the mass per sieve
    return 100*np.cumsum(data[::-1])[::-1]/np.sum(data)

def get_cumdist_array(DATA):
    # DATA - 2D array. Rows: Time, Columns: Mass ditribution
    return 100*np.fliplr(np.cumsum( np.fliplr(DATA), axis=1) / np.sum(DATA, axis=1)[:,None])

#------------------------------------------------------------------------
### Get the P statistics from the data sets or individual sream       ###
#------------------------------------------------------------------------
def get_pval(pval, mesh, data):
    # Get the P val for a single input distribution
    p_func = interpolate.interp1d(get_cumdist(data), mesh, fill_value="extrapolate")
    return p_func(pval)

def get_pval_array(pval, mesh, DATA):
    rows, cols= np.shape(DATA)
    p_vals = []
    for i in range(0, rows):
        p_func = interpolate.interp1d(get_cumdist(DATA[i]), mesh, fill_value="extrapolate")
        p_vals.append(p_func(pval))
    return np.array(p_vals)

#------------------------------------------------------------------------
### Creation functions                                                ###
#------------------------------------------------------------------------
def create_fracdist(mesh, Beta_param, D63_2_param):
    # create a fractional distribution. can be array or single
    if isinstance(Beta_param, (list, np.ndarray)):
        n = len(Beta_param)
    else:
        Beta_param = np.array([Beta_param])
        D63_2_param = np.array([D63_2_param])
        n = 1
    
    cum_dist = []
    for i in range(0, n):
        # Cumalative dist
        tmp = RR(mesh, Beta_param[i], D63_2_param[i])

        # fractional dist
        tmp1 = np.zeros(len(tmp))
        tmp1[0:-1] = tmp[0:-1] - tmp[1:]
        tmp1[-1] = tmp[-1]
        
        cum_dist.append( tmp1 )
    return np.reshape(np.array(cum_dist), (n,len(mesh)))

def RR(X, Beta, D63_2):
    # Roslin Rammler Distribution function
    # -------------------------------------------------------------------
    # PARAMETERS #
    # X     - Sieve mesh
    # Beta  - Slope parameter
    # D63_2 - D63 parameter, mesh size where 63% of material passes
    # OUTPUTS #
    # RR    - Size distribution as a fraction
    
    eps = X/X[0]
    n = np.divide(eps, (1 - 0.999*eps))
    n63_2 = ( D63_2/X[0] )/( 1 - D63_2/X[0] )

    R = 1 - np.exp( -1*(np.divide(n, n63_2)**Beta) )
    return R 
