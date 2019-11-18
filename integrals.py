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

Requirements: numpy, random, math, trajectory

Date            Author              Description
10/24/2019      Pierre Gauvreau     initial version - latin hypercube
11/17/2019      Pierre Gauvreau     uniform and gaussian distributions
"""

import numpy as np 
import random as r 
import math as m 
from trajectory import H

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
    
    return 2*x_diff*p_diff/(m.pi*(x_diff**2 + p_diff**2)**2)

def MC_mean_value(f, E_i, H_r, x_range, p_range, N, random_gen=None):
    """
    Computes the integral of f over a rectangular domain in phase space 
    defined by (x_range, p_range) using the mean value Monte Carlo method 
    with N samples.

    Parameters
    ----------
    f : function
        The integrand
    E_i : ndarray
        The reference point of the integrand in phase space
    x_range : ndarray
        The x boundaries of the domain of integration
    p_range : ndarray
        The p boundaries of the domain of integration
    N : int
        The number of samples to be used
    H_r : float
        The reference level of the trajectory
    random_gen : str
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

    gen_dict = {'uniform': gen_uniform, 'gaussian': gen_gaussian,
                'hypercube': gen_hypercube, None: gen_gaussian}
    
    if random_gen is not None:
        random_gen = random_gen.lower()
        assert random_gen in gen_dict.keys()

    V = (x_range[1]-x_range[0])*(p_range[1]-p_range[0])
    integral = 0

    for E in gen_dict[random_gen](x_range, p_range, N):
        if H(E) >= H_r:
            integral += winding_integrand(E_i, E)
    
    return V*integral/N 

    

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