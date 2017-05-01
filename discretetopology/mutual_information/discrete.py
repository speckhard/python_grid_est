#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 04:39:06 2017

@author: Daniel Speckhard, DTS@stanford.edu

This module contains methods relating to calculating the mutual information
between two distributions using the discrete mutual information formula. There
are two methods here. 1) Caclulate the discrete mutual 
information using methods from SK.learn. 2) Use maximum
likelihood estimators for mutual infromation written by the Weissman group at
Stanford, which in effect calculates the mutual information using the 
well-known discrete formula. 
"""

from sklearn.metrics import mutual_info_score
import est_MI as jvhw
import numpy as np

def find_sk_discrete_mi(vmag_matrix, bins):
        
    """ Finds the mutual information using discrete formula.
        
    This function calculates the mutual information between all the sets of
    two distinct nodes using the discrete formula for mutual information. 
        
    Args
    ----------
    vmag_matrix: ndarray, shape (# of data-ponts, # of nodes)
        The matrix contains the voltage magntiude data or the
        change in voltage magntiude data (if take_deriv() method has 
        been called).
            
    bins: int
        This controls how many bins to bin the continuous data
        from self.vmag_matrix. We must discretize the data before 
        taking the discrete mutual information. This parameter will 
        control the computation time of this function as well as the 
        precision.
    
    Returns
    ----------
    mi_matrix: 2D ndarray, shape (# of nodes, # of nodes)
        The matrix will be non-zero only for lower triangular values 
        since MI is symmetric and self-information values are of no use
        to us. The matrix will be calculated from the formula Sum_ij 
        p(i,j)log(p(i,j)/(p(i)*p(j))). """
            
    num_buses = len(vmag_matrix[1,:])
    mi_matrix = np.zeros((num_buses, num_buses))
    for i in range(1, num_buses):
        for j in range(0, i):
            mi_matrix[i, j] = _calc_discrete_mi(
                    vmag_matrix[:, i], vmag_matrix[:, j], bins)
    return mi_matrix

def _calc_discrete_mi(first_vec, second_vec, bins):
    """ Calculates discrete mutual information between two nodes.
            
    This function is a shell function to call sklearn's mutual info
    score function to calculate the mutual information between one pair
    of nodes. 
            
    Args
    ----------
    first_vec: 1D ndarray, shape (# of datapoints)
        This is the vector of voltage magnitude data for one
        node.
                
    second_vec: 1D ndarray, shape (# of datapoints)
        This is the vector of voltage magntiude data for 
        the second node. Both the first and second node form the 
        pair of nodes for which we calculate the mutual 
        information. Both vectors should be the same size.
                
    bins: int
        This controls how many bins to bin the ~continuous 
        voltage magnitude data. More bins means more precision in 
        the discretization. 
                
    Returns
    ----------
    mutual_info: double
        The mutual information, in bits, between the 'first'
        and 'second' node calculated using the discrete mutual
        information formula. 
    """
    c_xy = np.histogram2d(first_vec, second_vec, bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=c_xy)
    return mutual_info

def find_mle_mi(vmag_matrix):
    """ Finds the MLE mutual information of node pairs. 
        
    This function uses MLE entropy estimators to cacluate mutual
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
            mi_matrix[i, j] = jvhw.est_MI_MLE(
                    vmag_matrix[:, i], vmag_matrix[:, j])
    return mi_matrix