"""
We store functions for solving the winding number integral in this module.

General approach:
Given a contour Y and a data point E_i = (x_i, p_i) in phase space, the
winding number w(E_i, Y) is the number of times Y winds around E_i and is 
given by the line integral along Y of
    (1/2pi)*(-(p-p_i)dx + (x-x_i)dp)/((x-x_i)^2 + (p-p_i)^2)
We use green's theorem to convert this integral to the integral over the 
area enclosed by Y:
    (1/2pi)*(4(x-x_i)(p-p_i))/((x-x_i)^2 + (p-p_i)^2)^2 dxdp
If w(E_i, Y) = 1, then E_i is enclosed by Y. If w(E_i, Y) = 0, then E_i 
is not enclosed by Y.

Requirements: math, trajectory, random_generators

Date            Author              Description
10/24/2019      Pierre Gauvreau     initial version - latin hypercube
11/17/2019      Pierre Gauvreau     uniform and gaussian distributions
11/20/2019      Pierre Gauvreau     random number generators in separate
                                    module
01/25/2020      Keirn Munro         Mild refactor and renaming
"""

import math as m 
from trajectory import H
from random_generators import PointGenerators


def winding_integrand(E_i, E):
    """
    The transformed winding number function to be integrated over
    the domain enclosed by the trajectory.
    
    Parameters
    ----------
    E_i : ndarray
        The reference point about which we compute the winding
        number
    E : ndarray
        A point in phase space
    
    Returns
    -------
    float
        The value of the integrand at E
    """

    x_diff, p_diff = E_i[0] - E[0], E_i[1] - E[1]
    
    return 2 * x_diff * p_diff / (m.pi * (x_diff ** 2 + p_diff ** 2) ** 2)


def monte_mean_value(D, E_i, H_r, x_range, p_range, n, random_gen):
    """
    Computes the winding number about each E_i for points in a rectangular
    domain in phase space. Bounds are defined by (x_range, p_range) and points
    are generated by random_gen. Computation occurs using the mean value Monte
    Carlo method with N samples.

    Parameters
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E_i : ndarray
        The reference point of the integrand in phase space
    x_range : ndarray
        The x boundaries of the domain of integration
    p_range : ndarray
        The p boundaries of the domain of integration
    n : int
        The number of samples to be used
    H_r : float
        The reference level of the trajectory
    random_gen : Callable
        Optional: the random number distribution to be used. Currently
        supports 'uniform', 'gaussian', 'hypercube'. If none specified, 
        defaults to 2D uniform

    Returns
    -------
    float
        The value of the integral over the specified domain
    """

    assert x_range[1] > x_range[0]
    assert p_range[1] > p_range[0]
    v = (x_range[1] - x_range[0]) * (p_range[1] - p_range[0])
    random_gen = random_gen or PointGenerators.GAUSSIAN
    integral = 0

    for e in random_gen(x_range, p_range, n):
        if H(D, e) >= H_r:
            integral += winding_integrand(E_i, e)
    return v * integral / n
