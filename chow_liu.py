#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 07:18:33 2017

@author: Daniel Speckhard, dts@stanford.edu

This module contains methods to use the chow-liu algorithm to estimate the grid
topology.
"""

import networkx as nx
import numpy as np

def find_mst(mi_matrix):
        
    """ Finds the minimum spanning tree based on mutual information.
        
    This function finds the minimum spanning tree for the network using
    the mutual information values between nodes as weights. The function 
    acts a shell function to call networkx functions to find the min. span 
    tree.
        
    Args
    -------
    mi_matrix: 2D ndarray, shape (# of nodes, # of nodes)
        The mutual information matrix containing pair-wise
        mutual information for different nodes in the system. 
            
    Returns
    -------
    est_branches: 2D ndarray, shape (# of nodes - 1, 2)
        The estimated branches by running the minimum spaning
        tree algorithm using the mutual information values between 
        nodes as weights.
            
    graph: NetworkX graph object
        This is used for plotting/drawing the estimated graph. 
        """

    # We create a fully connected graph with weights specified by the
    # the mutual information values.
    # However, we have to take the reciprical of the MI matrix values
    # since nx only has min_span_tree as a fxn for python 2.7. We
    # add a tiny number to each value so reciprocal is defined.
    mi_matrix = mi_matrix + 1e-12
    # We cretae a temporary nx. graph object that's a cyclical graph
    # with edge weights equal to MI values in the matrix.
    net = nx.from_numpy_matrix(np.reciprocal(mi_matrix),
                               create_using=nx.Graph())
    mst = nx.minimum_spanning_tree(net)
    est_branches = np.array(mst.edges())
    graph = mst
    
    return est_branches, graph
