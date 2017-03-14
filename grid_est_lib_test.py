 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 05:22:47 2017

@author: DTS
This script contains unit tests for methods in grid_estimation_lib.py,
a library used for estimating grid topology.
"""

import numpy as np
from grid_estimation_lib import grid_est
import networkx as nx
import matplotlib.pyplot as plt
import graphviz as gv

# mst test
#true_branches = np.array(([3,1],[3,2]))
#vmag_data = np.array(([1,1,1],[1,1,1],[1,1,1],[1,1,1]))
#graph = grid_est(true_branches, vmag_data)
#graph.MI_matrix = np.array(([0,0,0], [2,0,0], [ 3, 4, 0]))
#
## Print MI_Matrix
#graph.MI_matrix = graph.MI_matrix + 1e-12
#print('MI matrix data')
#print(graph.MI_matrix)
#
## Now let's find the mst
##graph.find_mst()
#print(np.reciprocal(graph.MI_matrix))
#net = nx.from_numpy_matrix(np.reciprocal(graph.MI_matrix), 
#                                          create_using=nx.Graph())
###print(net.number_of_nodes())
###print(net.number_of_edges())
###print(net.nodes())
##print(net.edges())
###nx.draw(net)
##
###print(nx.adjacency_matrix(net, nodelist=None, weight='weight'))
#mst = nx.minimum_spanning_tree(net)
#print(mst.nodes())
#print(mst.edges())
#nx.draw(mst) #, weight = graph.MI_matrix)
#G = nx.cycle_graph(4)
#G.add_edge(0, 3, weight=2)
#T = nx.minimum_spanning_tree(G)
#sorted(T.edges(data=True))
#
#graph.find_mst()
#print(graph.est_branches)

# Load Data
SG_60min = np.genfromtxt('/Users/Dboy/Downloads/SG_data_solar_60min.csv',
                      skip_header = 9, delimiter = ',')

SG_60min = SG_60min[:,22:74]

# Import True Branches Data
true_branches = np.genfromtxt('/Users/Dboy/Downloads/SG1_true_branches.csv',
                              delimiter = ',')

sg1 = grid_est(true_branches, SG_60min)
#sg1.collapse_redundant_data()
#print(np.shape(sg1.vmag_matrix))
#print(sg1.num_buses)
#entropy_vec2 = sg1.find_gaussian_entropy()
#print(entropy_vec)
#joint_entropy_matrix = sg1.find_joint_gaussian_entropy()
sg1.run_mst('JVHW')
print(sg1.find_SDR())

#print(sg1.vmag_matrix)
#entropy_vec = np.zeros((sg1.num_buses))
#print('entropy vec')
#for i in range(0,sg1.num_buses):
#        # We use the equation H(X) = k/2*(1+ln(2*pi)) + 1/2*ln|Sigma|. 
#           # Where k is the dimension of the vector X and Sigma is the 
#           # covariance matrix. Note, here we only have one dimension, 
#           # vmag(i), in our entropy calc. This is not the case if we would 
#           # like to include phase information into our calculation.
#   #rint(i)
#   entropy_vec[i] = 0.5*(1+np.log(2*np.pi))+0.5*np.log(np.var(sg1.vmag_matrix[:,i]))
#                       