"""
We store custom random number generators in this module.

Requirements:
random, numpy

Date            Author              Description
11/20/2019      Pierre Gauvreau     initial version - uniform, gaussian,
                                    latin hypercube
"""

import numpy as np 
import random as r

def gen_uniform(x_range, p_range, N):
    """
    Generates uniform random values of (x, p) in the rectangle specified
    by (x_range, p_range) in phase space.

    Parameters
    ----------
    x_range : ndarray
        The x boundaries of the rectangle
    p_range : ndarray
        The p boundaries of the rectangle
    N : int
        The number of values to be generated

    Yields
    ------
    ndarray
        A point in phase space
    """

    x0, x1, p0, p1 = x_range[0], x_range[1], p_range[0], p_range[1]
    x_dist, p_dist = x1-x0, p1-p0

    for _i in range(N):
        x, p = x0 + x_dist*r.random(), p0 + p_dist*r.random()
        yield np.array([x, p])

def gen_gaussian(x_range, p_range, N):
    """
    Generates gaussian random values of (x, p) in the rectangle specified
    by (x_range, p_range) in phase space.

    Parameters
    ----------
    x_range : ndarray
        The x boundaries of the rectangle
    p_range : ndarray
        The p boundaries of the rectangle
    N : int
        The number of values to be generated

    Yields
    ------
    ndarray
        A point in phase space
    """

    x0, x1, p0, p1 = x_range[0], x_range[1], p_range[0], p_range[1]
    x_dist, p_dist = x1-x0, p1-p0

    mu_x, mu_p = x0 + x_dist/2, p0 + p_dist/2
    sigma_x, sigma_p = 1/3, 1/3 # I will make this user-specifiable later
    
    for _i in range(N):
        x, p = mu_x + mu_x*r.gauss(0, sigma_x)%1, mu_p + mu_p*r.gauss(0, sigma_p)%1
        yield np.array([x, p])

def gen_hypercube(x_range, p_range, N):
    """
    Generates random values of (x, p) from a 2D uniform latin hypercube
    in phase space.

    Parameters
    ----------
    x_range : ndarray
        The x boundaries of the hypercube
    p_range : ndarray
        The p boundaries of the hypercube
    N : int
        The number of values to be generated

    Yields
    ------
    ndarray
        A point in phase space
    """

    x0, x1, p0, p1 = x_range[0], x_range[1], p_range[0], p_range[1]
    x_step, p_step = (x1-x0)/N, (p1-p0)/N
    x_samples = [x0 + k*x_step + x_step*r.random() for k in range(N)]
    p_samples = [p0 + k*p_step + p_step*r.random() for k in range(N)]

    assert len(x_samples) > 0
    assert len(p_samples) > 0

    while len(x_samples) > 0:
        x_idx = r.randint(0, len(x_samples) - 1)
        p_idx = r.randint(0, len(p_samples) - 1)
        yield np.array([x_samples.pop(x_idx), p_samples.pop(p_idx)])

# Testing

if __name__ == '__main__':
    import pylab as pl 

    # test hypercube
    x, p = np.array([0, 2]), np.array([0, 1])
    N = 10
    x_list, p_list = [], []
    for s in gen_hypercube(x, p, N):
        x_list.append(s[0])
        p_list.append(s[1])
    pl.plot(x_list, p_list, '.')
    pl.show()