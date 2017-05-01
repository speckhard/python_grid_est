#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 04:39:06 2017

@author: Daniel Speckhard, DTS@stanford.edu

This module contains methods relating to calculating the mutual information
between two distributions using entropy estimators developed by the Weissman
Group at Stanford University.
"""

import est_MI as jvhw
import numpy as np

def find_jvhw_mi(vmag_matrix):

    """ Finds the mutual information of node pairs with JVHW estimators. 
        
    This function uses the JVHW entropy estimators to cacluate mutual
    information between nodes. This function acts as a shell to call
    a script written by the Weissman group to calculate mutual information
    between a pair of nodes. 
        
    Args
    ----------
    vmag_matrix: 2D ndarray, shape (# of data-ponts, # of nodes)
        The matrix contains the voltage magntiude or the change in 
        voltage magntiude (if the method take_deriv() has been called).
            
    Returns
    ----------
    mi_matrix: 2D ndarray, shape (# of nodes, # of nodes)
        The matrix will be non-zero only for lower triangular values 
        since MI is symmetric and self-information values are of no use
        to us.  The matrix will be calculated using JVHW estimators for
        entropy.
    """
    num_buses = len(vmag_matrix[1,:])
    mi_matrix = np.zeros((num_buses, num_buses))
    for i in range(1, num_buses):
        for j in range(0, i):
            mi_matrix[i, j] = jvhw.est_MI_JVHW(
                    vmag_matrix[:, i], vmag_matrix[:, j])
    return mi_matrix