"""
Main script for Hamiltonian Clustering.

Date        Author          Description
11/20/2019  GI              V1
"""

# default modules
import numpy as np
import pandas as pd
import os

# custom modules
import trajectory
from ODESolve import leapfrog
import integrals

# obtain data directory
data_dir = hc_config['data_dir']

# obtain data set for clustering
if hc_config['data_type'] == 'csv':
    data = pd.read_csv(data_dir) # read in assuming csv format

if hc_config['data_type'] == 'toy':
    data = 'toy' # select synthetic toy data set

# obtain trajectory parameters
data_copy = data.copy() # create copy of original data set
E_initial = np.array(data_copy.sample(n=1)) # take single data point of copied set

# begin contour computation by:
# - obtaining H_r and tuning parameters from user
# - obtain points upon S_i until contour is created by x_list & p_list
x_list, p_list = trajectory.compute_trajectory(E_initial, data, 
H_r, h, k1, k2, delta1, delta2, maxcount=5)

