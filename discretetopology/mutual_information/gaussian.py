#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 04:39:06 2017

@author: Daniel Speckhard, DTS@stanford.edu

This module contains methods relating to calculating the mutual information
between two distributions when we approximate the distributions as gaussian
distributions.
"""

import numpy as np

def find_gaussian_entropy(vmag_matrix):
        
    """ Calculates Gaussian entropy at a single node.
        
    This function finds the entropy by approximating the voltage
    magnitude data at different nodes as a Gaussian distribution.
        
    Args
    ----------
    self.vmag_matrix: ndarray, shape (# of data-points, # of nodes)
        This is the voltage magnitude data at different nodes and 
        timepoints.
                
    Returns
    ----------
        entropy_vec: ndarray, shape (# of nodes)
            Returns the entropy of each node with values of the entropy 
            vector (v(i)) equal to the entropy at Node i. The entropy is 
            caluclated by approximating the data at a node in the input 
            data matrix, self.vmag_matrix[:,i], as a Gaussian distribution.
    """
    num_buses = len(vmag_matrix[1,:])
    entropy_vec = np.zeros((num_buses))
    for i in range(0, num_buses):
        # We use the equation H(X) = k/2*(1+ln(2*pi)) + 1/2*ln|Sigma|.
        # Where k is the dimension of the vector X and Sigma is the
        # covariance matrix. Note, here we only have one dimension,
        # vmag(i), in our entropy calc. This is not the case if we would
        # like to include phase information into our calculation.
        #print(i)
        entropy_vec[i] = 0.5*(1+np.log(2*np.pi))+ \
                   0.5*np.log(np.var(vmag_matrix[:, i]))
        #print(entropy_vec[i])

    return entropy_vec

def find_joint_gaussian_entropy(vmag_matrix):
    """ Calculates joint Gaussian entropy between two nodes.

    This function calculates the pair-wise mutual information between
    two nodes in the input data matrix, node_volt_matrix. The mutual
    information is claculated by appoximating the distribution of data at
    each node as a Gaussian distribution.

    Args
    ----------           
    vmag_matrix: ndarray, shape (# of data-points, # of nodes)
        A matrix containing the voltage magnitude data.
        The number of columns should be the number of nodes in the system. 
        The number of rows in the matrix corresponds to the number of 
        datapoints where the voltage magnitude is measured across all 
        nodes.

    Returns
    ----------
    joint_entropy_matrix: ndarrray, shape (# of nodes, # of nodes)
        This matrix contains the joint entropy between two nodes from 
        the input matrix, self.vmag__matrix. The matrix is a lower 
        triangular matrix since the joint entropy is symmetric, i.e., 
        Entropy(i,j) = Entropy(j,i). The joint entropy of the same 
        node, Entropy(i,i), is not used in the chow-liu algorithm 
        and therefore these values (the diagonal of 
        joint_entropy_matrix) are set to zero. The entropy is 
        calculated by approximating the data at a node in the input 
        data matrix, node_volt_matrix(:,i), as a Gaussian distribution.
        We use the equation joint_entropy([X,Y]) = k/2*(1+ln(2*pi)) + 
        1/2*ln|Sigma|. Where k is the dimension of the vector [X,Y] 
        and Sigma is the covariance matrix.
        """
    num_buses = len(vmag_matrix[1,:])
    # Initialize the joint entropy matrix, note the size it determined
    # by the number of buses in the grid.
    joint_entropy_matrix = np.zeros((num_buses, num_buses))

    # We avoid calculating the joint_entropy values using the same node
    # twice. Therefore the diagonal values for the join_entropy_matrix
    # are not calculated.
    for i in range(1, num_buses):
        for k in range(0, i):

            # We use the equation joint_entropy([X,Y]) = k/2*(1+ln(2*pi)) +
            # 1/2*ln|Sigma|. Where k is the dimension of the vector [X,Y] and
            # |Sigma| is the deteriminant of the covariance matrix. For two
            # nodes, where each node contributes it's voltage magntiude data,
            # k, the dimension of [X,Y] is equal to two.

            det_of_cov_matrix = np.linalg.det(np.cov(
                    vmag_matrix[:, i], vmag_matrix[:, k]))

            # Let's check if the determinant value is very close to zero
            # and negative. If this is true, a numerical rounding error has
            # likley happened and we want to avoid getting a log of a negative
            # number.
            if (det_of_cov_matrix <= 0) and (det_of_cov_matrix > -0.0001):
                # Then we assume this is a numerical error and we make the
                # mutual information negative by setting the joint entropy
                # value to a very negative number. This is abritrary and this
                # case should never occur in practice unless we end up
                # computing the joint entropy of two nodes that are labelled
                # differently but contain the same data.
                joint_entropy_matrix[i, k] = -1E3
            else: # Othewrwise we compute as normal.
                joint_entropy_matrix[i, k] = 2/2*(
                        1+np.log(2*np.pi))+0.5*np.log(det_of_cov_matrix)
    return joint_entropy_matrix

def find_mi_mat(entropy_vec, joint_entropy_matrix):
        
    """ Finds the mutual information from between nodes using entropy vals. 
        
    The function takes in as input the single node entropy values
    (H(i)) and joint entropy matrix values (H(i,j)). The output is a lower
    triangular matrix that contains the mutual information MI(i,j). Since
    mutual information is symmetric we only have to calculate the lower
    triangular values.
        
    Args
    ----------
    Single_node_entropy_vec: This is a vector size (1 x number of
        buses) with values of the vector, v(i) equal to the entropy at
        node(i). The entropy is caluclated by approximating the data at a
        node in the input data matrix, node_volt_matrix(:,i), as a Gaussian
        distribution.
            
    joint_entropy_matrix: This matrix contains the joint entropy
        between two nodes. The matrix is a lower triangular matrix since 
        the joint entropy is symmetric, i.e., Entropy(i,j) = Entropy(j,i).
        The joint entropy of the same node, Entropy(i,j) is not used in the
        chow-liu algorithm and therefore these values (the diagonal of
        joint_entropy_matrix) are set to zero. % The entropy is caluclated 
        by approximating the data at a node in the input data matrix,
        node_volt_matrix(:,i), as a Gaussian distribution. The size of the
        matrix is (number of nodes x number of nodes). We use the
        equation joint_entropy([X,Y]) = k/2*(1+ln(2*pi)) + 1/2*ln|Sigma|. 
        Where k is the dimension of the vector [X,Y] and Sigma is the 
        covariance matrix.
            
    Returns
    ----------
    mi_matrix: This matrix is size (number of nodes x number of
        nodes). The matrix is lower triangular, meaning the diagonal and
        elements above the diagonal are zero since the mutual information
        between two nodes MI(i,j) = MI(j,i) is symmetric and the
        self-information MI(i,i) is not used in the algorithm since we 
        don't connect a node to itself (to avoid cycles). The mutual 
        information can be caluclated from the joint entropy and the single
        node entropy by recalling MI(i,j) = entropy(i) +  entropy(j) - 
        joint_entropy(i,j). """

    num_buses = len(joint_entropy_matrix[1,:])
    mi_matrix = np.zeros((num_buses, num_buses))
    for i in range(1, num_buses):
        for k in range(0, i):
            # The mutual information can be caluclated from the joint entropy
            # and the single node entropy by recalling MI(i,j) = entropy(i) +
            # entropy(j) - joint_entropy(i,j).
            mi_matrix[i, k] = entropy_vec[i] + entropy_vec[k] - \
                          joint_entropy_matrix[i, k]
    return mi_matrix