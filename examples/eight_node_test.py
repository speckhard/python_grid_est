#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:43:21 2017

@author: Daniel Speckhard dts@stanford.edu

This script acts as an example to run the mutual information based grid
estimation aglorithm developed in the Sustainable Systems Lab at Stanford
University on a 8 bus IEEE network. 
"""
import numpy as np
from scipy.io import loadmat
from estimation.grid_est import GridEst

# 8 Node Example.
matfile = loadmat('/Users/Dboy/Downloads/Node8_randPF_solar.mat',
                  squeeze_me=True, struct_as_record=False)

# Load the voltage magnitude matrix.
vmag_matrix = matfile['v_vec']
# Remove the feeder Node 1 data from v_vec.
vmag_matrix = np.delete(vmag_matrix, 0, 1)
# Load the ture branches of the network.
mpc_base = matfile['mpc_base']

# Take the columns corresponding to true branches.
# Subtract by 1, to account for feeder node being deleted.
true_branches = mpc_base.branch[:,0:2]
# Remove the feeder branch (1-2) from the true branches array.
# Subtract by 1, to account for feeder node being deleted.
true_branches = np.delete(true_branches, 0, 0 ) - 1
# Create a GridEst objec to prepare to run the algorithm.
node8_randPF_solar = GridEst(true_branches, vmag_matrix, 'Node8_randPF_solar')
# Choose mutual information method, either gaussian, MLE, sk_discrete, or JVHW.
mi_method = 'MLE'
node8_randPF_solar.run_mi_alg(mi_method)
# Print successful detection rate of algorithm.
print('The SDR is: ' + str(node8_randPF_solar.find_sdr()[0]) + '%')