# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:58:44 2022

@author: Dan
"""
import numpy as np

def fd_coeff(z,x,k):
    '''
    
    Parameters
    ----------
    z : Float
        Location to estimate derivative.
    x : Array
        1D array of datapoints for function.
    k : Positive integer
        Highest order of derivative to estimate.

    Returns
    -------
    C : Array 
        2D array of derivative at datapoints,
        i-th column is for the i-th order derivative.

    '''
    
    n = len(x) - 1
    
    if k > n:
        raise ValueError("Need at least k or more datapoints to estimate k-th derivative")
    
    c1 = 1
    c4 = x[0] - z
    C = np.zeros((n+1,k+1))
    C[0,0] = 1
    for i in range(1,n+1):
        mn = min(i,k)
        c2 = 1
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2*c3
            if j==i-1:
                for s in range(mn,0,-1):
                    C[i,s] = c1*(s*C[i-1,s-1] - c5*C[i-1,s])/c2
                C[i,0] = -1*c1*c5*C[i-1,0]/c2
            for s in range(mn,0,-1):
                C[j,s] = (c4*C[j,s] - s*C[j,s-1])/c3
            C[j,0] = c4*C[j,0]/c3
        c1 = c2
    
    return C




