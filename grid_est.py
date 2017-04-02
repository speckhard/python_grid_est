#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:21:28 2017

@author: Daniel Speckhard, DTS@stanford.edu

This library was created at the Stanford Sustainable Systems Lab. This library
includes functions to estimate the topology of an electrical grid given voltage
measurements at nodes.
"""
import numpy as np
import networkx as nx
import redundant_data as rd
import mutual_info as mi
import transform_data as td
import chow_liu as cl
import matplotlib.pyplot as plt

class GridEst(object):


    """ Class to statistically estimate topology of electrical grid.
    
    GridEst is a class containing variables and methods to estimate
    the topology of system using a variety of statistical methods from voltage
    magnitude data taken at each node in the network.
    
    Currently, only the Chow-Liu algorithm has been implemented. In the future
    a LASSO method will be implemented. 
    """
    
    def __init__(self, true_branches, vmag_matrix, graph_name):

        """ Initialize grid est class instance.

        Args
        ----------
        true_branches: ndarray, shape (# of nodes - 1, 2)
            This is matrix storing the node lables of each pair of nodes
            connected by a true branch in the true tree structure. Ex. [(2,1),
            (3,1)] would indicate that nodes 2 and 3 are both connected to
            Node 1.

        vmag_matrix: ndarray, shape (# of datapoints, # of nodes)
            This matrix should contain the voltage magnitude
            data at different timepoints and different nodes. The matrix should
            be size ((number of nodes) x (number of data points at each node)).
            
        graph_name: str
            This is the name of the network. When we plot the estimated newtork
            we will save a figure with this name. 

        Returns
        ----------
        self.num_buses: int
            This is the number of buses in the network.

        self.mi_matrix: str
            This variable is initialized to a string labeled undefined so that
            the user is sure that the values have not yet been calculated. When
            the values are calculated self.mi_matrix should become an ndarray
            shape (# of nodes, # of nodes).

        self.est_branches: ndarray, shape (# of nodes - 1, 2)
            These are the branches that are estimated by the algorithm. The
            values are intialized to zero (bogus values) and should be filled
            by calling appropriate methods.

        self.graph: str
            This variable should eventually store the Networkx object graph
            ouput from estimation. The variable is initialized to 'undefined'
            so that the user knows the estimation has not yet taken place.
        
        self.graph_name: str
            The name of the graph which will be estimated. 

        """
        self.true_branches = true_branches -1
        self.vmag_matrix = vmag_matrix
        self.mi_matrix = 'undefined' # Leave this undefined until later.
        self.num_buses = len(self.vmag_matrix[0, :])
        self.est_branches = np.zeros((self.num_buses - 1, 2))
        self.graph = 'undefined'
        self.graph_name = graph_name


    def find_sdr(self):

        """ Find the succesfull detection rate of estimating branches.

        This function returns the percentage of estimated branches which are
        true branches (branches contained in self.true_branches).

        Args
        ----------
        self.est_branches: 2D ndarray, shape (number of nodes - 1, 2)
            Each row is a pair of nodes corresponding to an estimated branch
            from the algorithm output. The number of branches should be one
            less than the total number of nodes since we are estimating a tree.

        self.true_branches: 2D ndarray, shape (number of nodes - 1, 2)
            This is a matrix containing the true branches in the system.
            The number of rows in this matrix corresponds to the number of
            true branches in the system. The number of columns should be equal
            to two, which is the number of nodes involved in a single branch.

        Returns
        ----------
        successful_branch_counter: int
            This is a integer counter that tells us how many branches have been
            correctly estimated by the algorithm.

        sdr: double
            This value is the successful detection rate (SDR) percentage of
            the estimation algorithm. It equals the number of correctly
            identified branches divided by the total number of branches.
        """

        successful_branch_counter = 0.0
        #flipped_est_branches = np.fliplr(self.est_branches)
        for i in range(0, np.shape(self.est_branches)[0]):
            if ([self.est_branches[i, 0], self.est_branches[i, 1]] \
            in self.true_branches.tolist()) or ([self.est_branches[i, 1],
                                                 self.est_branches[i, 0]] in
                                                self.true_branches.tolist()):

                successful_branch_counter += 1.

        sdr = np.divide(successful_branch_counter,
                        len(self.est_branches[:, 0]))
        # Multiply by 100 to get the percent
        sdr = 100.0*sdr

        return sdr, successful_branch_counter

    def run_chow_liu(self, mi_method):

        """ This method estimates the topology of network using Chow-Liu Alg.

        The chow-liu algorithm is run wherby a network is estimated using the
        minimum spanning tree with the mutual information between nodes used
        as weights. The input data is voltage magntiude data taken at
        each node at constant time intervals. The algorithm is run on the
        change in voltage magntiude between one time-point and the next for
        improved results.

        Args
        ----------
        self.vmag_matrix: ndarray, shape (# of datapoints, # of nodes)
            This matrix should contain the voltage magnitude
            data at different timepoints and different nodes. The matrix should
            be size ((number of nodes) x (number of data points at each node)).

        self.true_branches: 2D ndarray, shape (number of nodes - 1, 2)
            This is a matrix containing the true branches in the system.
            The number of rows in this matrix corresponds to the number of
            true branches in the system. The number of columns should be equal
            to two, which is the number of nodes involved in a single branch.

        self.graph: str
            This variable should eventually store the Networkx object graph
            ouput from estimation. The variable is initialized to 'undefined'
            so that the user knows the estimation has not yet taken place.
            
        mi_method: str
            This variable determines how to calculate the mutual information 
            between nodes. We can select between gaussian, sk_discrete, MLE or
            JVHW. The MLE and sk_discrete method should return the same values
            as both use the discrete mutual information formula to calculate 
            the mutual information, however, the implementations are different
            so there may be small differences. The gaussian method approximates
            the voltage magntiude data as a gaussian function. The JVHW method
            uses entropy estimators developed by the Weissman group at
            Stanford.

        Returns
        ----------
        A figure of the estimated graph is saved to the current directory as a
        PNG file.

        """
        self.vmag_matrix, self.true_branches  = rd.collapse_redundant_data(
                self.vmag_matrix,self.true_branches)
        # Take delta/deriv of data, with step-size/spacing of one data-point.
        deriv_step = 1
        self.vmag_matrix = td.take_deriv(self.vmag_matrix, deriv_step)
        # Find the mutual information
        bits = 8
        bins = 2**8

        if mi_method == 'gaussian':
            entropy_vec = mi.find_gaussian_entropy(self.vmag_matrix)
            joint_entropy_matrix = mi.find_joint_gaussian_entropy(
                    self.vmag_matrix)
            self.mi_matrix = mi.find_mi_mat(entropy_vec, joint_entropy_matrix)

        elif mi_method == 'sk_discrete':
            self.vmag_matrix = td.discretize_signal(self.vmag_matrix, bits)
            self.mi_matrix = mi.find_sk_discrete_mi(self.vmag_matrix, bins)
        
        elif mi_method == 'MLE':
            self.vmag_matrix = td.discretize_signal(self.vmag_matrix, bits)
            self.mi_matrix = mi.find_mle_mi(self.vmag_matrix)

        elif mi_method == 'JVHW':
            self.vmag_matrix = td.discretize_signal(self.vmag_matrix, bits)
            self.mi_matrix = mi.find_jvhw_mi(self.vmag_matrix)
            
        else:
            print('Please select a valid mutual information method' + 
                  ' gaussian, sk_discrete, MLE or JVHW')
            pass
        
        # Find the minimum spanning tree using mutual information as weights.
        self.est_branches, self.graph = cl.find_mst(self.mi_matrix)
        
        # 
        self.plot_graph('twopi')



    def plot_graph(self, graphviz_layout):
        """ Method to save newtorkx graph object as a dot file.

        This method plots the graph in self.graph using a graphviz layout
        program. The plot is saved as a png. This method is mostly a shell 
        function to call nx.draw_graphviz() to draw the newtorkx object.

        Args
        ----------
        self.graph: networkx graph object
            This object contains the estimated graph from the estimatio
            algorithm.
        
        self.graph_name: str
            This object will save the plotted estimated graph as a PNG file
            with this name. 

        graphviz_layout: string
            Speficied the layout of the graph when draw the graph onto a
            figure. We can choose from, twopi, twopi, gvcolor, wc, ccomps,
            tred, sccmap, fdp, circo, neato, acyclic, nop, gvpr, dot, sfdp.
            
        Returns
        ----------
        The graph in self.graph is drawn onto a matlabplot.pyplot figure and
        saved as a PNG file. 

        """

        # One alternative option is to save the graph as a dot file with:
        # nx.write_dot(self.graph,"grid.dot")
        
        plt.figure()
        nx.draw_graphviz(self.graph, graphviz_layout, with_labels = True,
                         node_color = 'r', node_size = 200, font_size = 10 )

        plt.savefig(self.graph_name + '.png', format='PNG')
        plt.show()
        pass

