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


def gen_uniform(x_range, p_range, num_pts):
    """
    Generates uniform random values of (x, p) in the rectangle specified
    by (x_range, p_range) in phase space.

    Parameters
    ----------
    x_range : ndarray
        The x boundaries of the rectangle
    p_range : ndarray
        The p boundaries of the rectangle
    num_pts : int
        The number of values to be generated

    Yields
    ------
    ndarray
        A point in phase space
    """
    for _i in range(num_pts):
        pt_x = r.uniform(*list(x_range))
        pt_p = r.uniform(*list(p_range))
        yield np.array([pt_x, pt_p])


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

    assert len(x_range) == len(p_range) == 2
    mu_x, mu_p = x_range[0] + np.diff(x_range).item() / 2, p_range[0] + np.diff(p_range).item() / 2
    sigma_x, sigma_p = 1 / 3, 1 / 3  # I will make this user-specifiable later
    
    for _i in range(N):
        x_pt, p_pt = mu_x + mu_x*r.gauss(0, sigma_x) % 1, mu_p + mu_p * r.gauss(0, sigma_p) % 1
        yield np.array([x_pt, p_pt])


def gen_hypercube(x_range, p_range, num_pts, rand_pt_gen=gen_uniform):
    """
    Generates random values of (x, p) from a 2D uniform latin hypercube
    in phase space.

    Parameters
    ----------
    x_range : ndarray
        The x boundaries of the hypercube
    p_range : ndarray
        The p boundaries of the hypercube
    num_pts : int
        The number of values to be generated
    rand_pt_gen : Callable[[ndarray, ndarray, int], ndarray]
        The random point generator to use for populating values

    Yields
    ------
    ndarray
        A point in phase space
    """
    x_step, p_step = (np.diff(x_range).item() / num_pts,
                      np.diff(p_range).item() / num_pts)
    unit_x_box = x_range[0], x_range[0] + x_step
    unit_p_box = p_range[0], p_range[0] + p_step
    rand_points = rand_pt_gen(unit_x_box, unit_p_box, num_pts)
    pt_range = list(range(num_pts))
    x_indices = r.sample(pt_range, k=num_pts)
    p_indices = r.sample(pt_range, k=num_pts)

    for i in range(num_pts):
        rand_x, rand_p = next(rand_points)
        yield np.array([rand_x + x_indices[i] * x_step,
                        rand_p + p_indices[i] * p_step])

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