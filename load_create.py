
"""
Functions to either create or load datasets

Guidelines:
Functions should output unordered sets of (x,p) ndarrays

Requirements:
numpy, pandas, random_generators

Date            Author              Description
01/25/2020      Pierre Gauvreau     initial verison
"""

import numpy as np
import pandas as pd
import random_generators as rg


def create_test_set(x_range, p_range, num_pts, num_groups):
    """
    Creates a randomly generated set of data points in a rectangle defined
    by x_range, p_range, "pre-clustered" for easy visual inspection of 
    algorithm results.

    Parameters
    ----------
    x_range : ndarray
        The x boundaries of the rectangle
    p_range : ndarray
        The p boundaries of the rectangle
    num_pts : int
        The total number of points to be generated
    num_groups : int
        The total number of test clusters to be generated

    Returns
    -------
    set(ndarray)
        Test data set
    """
    data_set = set()
    x_step, p_step = (np.diff(x_range).item() / num_groups,
                      np.diff(p_range).item() / num_groups)
    group_sizes = rg.gen_capacities(num_pts, num_groups)

    for group_center in rg.gen_hypercube(x_range, p_range, num_groups):
        group_x_range = np.array([
            min(group_center[0] - x_step / 2, x_range[0]),
            max(group_center[0] + x_step / 2, x_range[1])])
        group_p_range = np.array([
            min(group_center[1] - p_step / 2, p_range[0]),
            max(group_center[1] + p_step / 2, p_range[1])])
        data_set |= set(rg.gen_gaussian(group_x_range, group_p_range,
                                        next(group_sizes)))
    return data_set


# Testing
if __name__ == '__main__':
    import pylab as pl

    x, p = np.array([0, 2]), np.array([0, 2])
    D = create_test_set(x, p, 450, 3)
    x_list, p_list = zip(*D)

    pl.plot(x_list, p_list, '.')
    pl.show()
