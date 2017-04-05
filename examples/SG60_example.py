#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:54:39 2017

@author: Daniel Speckhard

Example to run SG1 solar 60 min
"""
import numpy as np
from grid_est import GridEst
# SG 60 min example 
SG_60min = np.genfromtxt('/Users/Dboy/Downloads/SG_data_solar_60min.csv',
                      skip_header = 9, delimiter = ',')

SG_60min = SG_60min[:,22:74]

# Import True Branches Data
true_branches = np.genfromtxt('/Users/Dboy/Downloads/SG1_true_branches.csv',
                              delimiter = ',')

sg1 = GridEst(true_branches, SG_60min, 'SG1_60min')
sg1.run_chow_liu('discrete')
print(sg1.find_sdr())
