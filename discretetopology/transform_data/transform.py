#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 26 05:51:21 2017

@author: Daniel Speckhard, dts@stanford.edu

This module contains methods to transform sensor data to improve
the efficiency or accuracy of the grid estimation algorithm when
run on the resulting transformed data.
"""
import numpy as np

def discretize_signal(vmag_matrix, bits):

    """ 
    Discretizes data with an input number of bits precision. 
        
    This function discretizes the voltage magnitude data. 
    self.vmag_matrix: A matrix containing the voltage magnitude data.
    The number of columns should be the number of nodes in the system. The
    number of rows in the matrix corresponds to the number of datapoints
    where the voltage magnitude is measured across all nodes.
        
    Args
    -------
    bits: int
        Exponential number of bins with which to discreteize data. 
        Meaning, if bins = 10, we bin the data into 2^10 bins. In this 
        scheme we make each bin equal size.
            
    Returns
    -------
        vmag_matrix: 2D ndarray, shape (# of data-points, # of nodes)
        This matrix is the size of the input
        vmag_matrix. The values in this matrix are all integers. 
        The values in the output version of the matrix are discretized 
        are equal to the bin number which the value of the input matrix
        falls into with respect to the binning process.
    """
    global_min = np.min(np.min(vmag_matrix))

    global_max = np.max(np.max(vmag_matrix))

    # Shift the data so new minimum is at zero.
    vmag_matrix = vmag_matrix - global_min
    # Determine the bin-size for equally spaced bins.
    bin_size = np.divide((global_max-global_min), (2**bits - 1))
    # Now find new values for data:
    vmag_matrix = \
        np.round(np.divide(vmag_matrix, bin_size)).astype(int) 
    return vmag_matrix
    
def get_delta(vmag_matrix, deriv_step):
        
    """ Convert voltage mag. data into the change in voltage mag. wrt time.  
        
    This function finds the change in voltage magntiude between 
    time-points. The spacing between time-points is a variable that can be
    changed. The change in voltage magnitude is calculated as Delta_Vmag =
    Vmag(node x, time t) - Vmag(node x, time t-deriv_step).
        
    Args
    -------
    vmag_matrix: 2D ndarray, shape (# of data-points, # of nodes)
        This is the voltage magnitude data at different nodes. We want to 
        transform this data so we end up with the change
        in voltage magnitude from one time-point to the next for each node.
        
    deriv_step: int 
        This parameter decides how many time-points for which
        to compute the voltage magntiude difference at a node. Ex. if
        deriv_step = 5, then the change in voltage magnitude will be
        Delta_Vmag = Vmag(node x, time t) - Vmag(node x, time t-5).
        
    Returns
    -------
    vmag_matrix: 2D ndarray, shape (# of data-points, # of nodes)
        This matrix now represents the change in voltage
        magnitude at a node from one time-point to the next. Each column
        corresponds to a different node and each row to a time-point.
    """
    end = len(vmag_matrix[:, 1])-1
    vmag_matrix = vmag_matrix[
            deriv_step+1:-1, :] - vmag_matrix[1:end-deriv_step, :]
    
    return vmag_matrix