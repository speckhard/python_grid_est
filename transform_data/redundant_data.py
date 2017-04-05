#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 25 02:32:07 2017

@author: Daniel Speckhard, DTS@stanford.edu

This module contains a methods to clean input voltage magntiude data and the 
true branch list of the network thereafter.
"""

import numpy as np

def collapse_redundant_data(vmag_matrix, true_branches):

    """
    Removes nodes in the network with identical data to other nodes.

    This function will check to see if there are any nodes that contain
    voltage measurements that are the same as another node. For instance,
    if 40 nodes are given, this function will check if Node 1 has the same
    data as Node 2. If Node 1 and Node 2 have the same data this
    function will collapse the two nodes, meaning it will remove Node 2
    and fix the list of true branches to replace any reference to Node 2
    with Node 1.

    Args
    -------
    vmag_matrix: 2D ndarray, shape (# of data-points, # of nodes)
        A matrix containing the voltage magnitude data.
        The number of columns should be the number of nodes in the system.
        The number of rows in the matrix corresponds to the number of
        datapoints where the voltage magnitude is measured across all
        nodes. When this function is called the matrix should not be empty.

    true_branches: 2D ndarray, shape (# of nodes - 1, 2)
        This is a matrix containing the true branches in
        the system. The number of rows in this matrix corresponds to the
        number of true branches in the system. The number of columns should
        be equal to two, which is the number of nodes involved in a single
        branch.

    Returns
    -------
    vmag_matrix: 2D ndarray, shape (# of data-points,
                                             # of unique nodes)
        This matrix is the voltage magnitude dataset with the redundant
        nodes removed. Therefore, the output matrix will be different size
        than the input matrix if redundant nodes are present in the input
        matrix. The word redundant here refers to a node that has
        all the same data values as another node. The node which is lower
        in number is retained when a redundant node is found. Ex. If Node
        34 has the same data as Node 1. Node 34 will be removed from node
        volt matrix.

    true_branches: 2D ndarray, shape (# of nodes - 1, 2)
        The list of true branches is also an output, it
        has redundant nodes replaced with lower node label number
        non-redundant nodes. Ex. Imagine Node 34 has the same data as Node
        1. Any branches in the true_branches matrix containing node 34,
        let's say a branch (7,34). The output true_branches will now list
        (7,1). If there's a branch from (1,34) it will have to be deleted
        since a branch (1,1) is a self-referring branch. Self-referring
        branches are sometimes returned by this function. This function
        does not remove self-referring branches.
    """
    num_buses = len(vmag_matrix[1,:])
    number_observations = len(vmag_matrix[:, 1])

    # We initialize a variable for the number of identical nodes found.
    # This is useful for debugging purposes.
    number_identical_nodes = 0

    # We will cycle through the node_volt_matrix and see if we can find
    # redundant nodes. If so we will delete this node. Note the number of
    # columns (i.e. nodes) is changing, this means we need to
    # evaluate this number every loop of the cycle.

    # We have to use a while loop since in Matlab for loops evaluate
    # bounds at first run and we will change the size of our matrix that
    # we iterate through, since we find identical pairs, we'll remove
    # columns.
    outer_loop_index = 0

    matrix_of_identical_nodes = np.zeros(((num_buses -1), 2))

    while outer_loop_index < (len(vmag_matrix[1, :]) -1):
        # We want to evaluate whether one node is the same as another
        # column so we check starting with node i+1.
        inner_loop_index = outer_loop_index + 1
        while inner_loop_index <= (len(vmag_matrix[1, :]) -1):
            # Check whether it is the same node.
            if np.sum(vmag_matrix[:, outer_loop_index] ==
                      vmag_matrix[:, inner_loop_index]) == \
                      number_observations:

                number_identical_nodes = number_identical_nodes + 1
                matrix_of_identical_nodes[number_identical_nodes, :] = \
                                             outer_loop_index, inner_loop_index
                # Now let's remove the identical nodes.
                vmag_matrix = \
                np.delete(vmag_matrix, inner_loop_index, 1)

                # Update the list of true branches.
                # Go through the rows of the true branch data set.
                for i in range(0, num_buses -1):
                # Check if we find the second identical node-k in the row.
                    for j in range(0, 2):
                        if true_branches[i, j] == inner_loop_index:
                            # set the higher number redundant node equal to the 
                            # lower number redundant node.
                            true_branches[i, j] = outer_loop_index
                        elif true_branches[i, j] > inner_loop_index:
                    # Since I've just deleted a col, I also need to
                    # downshift all the values in the True Branch Data set
                    # that were above this col. Ex. if i delete column 5
                    # (node 5), i need to make nodes 6 appear as node 5 in
                    # the true branch data.
                            true_branches[i, j] = \
                                         true_branches[i, j] -1

        # If a column was not deleted continue iterating. We don't
        # increment iterator if a col was deleted since now col k is a
        # different col.
            else:
                inner_loop_index = inner_loop_index +1
        outer_loop_index = outer_loop_index + 1

    # Update self.num_buses
    num_buses = len(vmag_matrix[1, :])
    return vmag_matrix, true_branches

def remove_redundant_branches(true_branches):
    """
    Removes branches that start at one node and end at the same node.

    This function removes self-referring branches from true_branches.
    Ex. if there is a branch in self.true_branches such as (1,1), it will
    be removed. Redundant branches (like (1,1)) should exist in
    self.true_branches only after the method collapse_redundant_nodes
    has been called.

    Args
    -------
    true_branches: 2D ndarray, shape (# of nodes - 1, 2)
        This is a matrix containing the true branches in
        the system. The method remove_redundant_branches is called when we
        suspect self.true_branches has additional self-referring branches.
        The number of rows in this matrix corresponds to the
        number of true branches in the system plus any self-referring
        branches. The number of columns should be equal to two, which is
        the number of nodes involved in a single branch.

    Returns
    -------
    true_branches: 2D ndarray, shape (# of nodes - 1, 2)
        This is a matrix containing the true branches in
        the system. The number of rows in this matrix corresponds to the
        number of true branches in the system since this method will have
        removed any self-referring (redundant) branches. The number of
        columns should be equal to two, which is the number of nodes
        involved in a single branch.
    """

    branch_counter = 0;

    # Check every branch.
    while branch_counter < len(true_branches[:,1]):
        # Check if both nodes in a single branch are equal to each other.
        if true_branches[branch_counter, 0] == \
                        true_branches[branch_counter,1]:
            # delete the row containing the self-referrring branch.
            np.delete(true_branches, branch_counter, 0)
        else: # increment counter so we can check the next branch.
            branch_counter = branch_counter + 1;

    return true_branches