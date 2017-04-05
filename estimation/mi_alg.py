#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Created on Tue Mar  7 22:21:28 2017

@author: Daniel Speckhard, DTS@stanford.edu

This module includes functions to estimate the topology of an electrical grid 
given voltage measurements at nodes using mutual information based
estimation.
"""

import networkx as nx
import numpy as np

def find_mst(mi_matrix):
        
    """ Finds the maximum spanning tree based on mutual information.
        
    This function finds the maximum spanning tree for the network using
    the mutual information values between nodes as weights. NetworkX's
    minimum spanning tree funciton is called on the recriprocal of the
    mutual information network, therefore generating the maximum
    spanning tree of the network.
      
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