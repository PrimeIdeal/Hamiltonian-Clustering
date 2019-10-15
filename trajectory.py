"""
We store functions for defining and solving the trajectories of the 
Hamiltonian system in this module.

General approach: given data set D = {E_1, E_2, ..., E_n} in R^2 where
E_i = (x_i, p_i), define the Hamiltonian function
    H(E) = sum(H_i(E)) for i in [1, N]
where
    H_i(E) = exp(-rho(E)^2)
    rho = ||E - E_i|| (Euclidean distance)
We select 0 < H_r < 1 and k1, k2 > 0, then pick E_i in D as an initial
condition to solve the modified dynamics
    dx/dt = f_1(x, p)(H_p - H_x(H - H_r)^1/3)
    dp/dt = f_1(x, p)(H_x + H_p(H - H_r)^1/3)
until we reach time t where H(E(t)) = H_r, where H_x, H_p are the partial 
derivatives of H with respect to x and p and
    f_1(x, p) = k1(H_x^2 + H_p^2)^-1
We then repeat the process, using E(t) as our initial condition and
replacing f_1(x, p) with
    f_2(x, p) = k2(H_x^2 + H_p^2)^-1/2
until we have computed a closed trajectory S.

Requirements: numpy, math, ODESolve

Date            Author              Description
10/15/2019      Pierre Gauvreau     initial version - leapfrog algorithm
"""


import numpy as np 
import math as m
from ODESolve import leapfrog

def euclidean(E1, E2):
    """
    Returns Euclidian distance between two points E1 and E2 in R^2.

    Parameters
    ----------
    E1 : ndarray
        Point in R^2
    E2 : ndarray
        Point in R^2

    Returns
    -------
    float
        Euclidean distance between the two points
    """
    return m.sqrt(np.dot(E1-E2, E1-E2))

def H_i(E, E_i):
    """
    Helper function for H. We construct a 2D gaussian about each
    observed data point E_i.

    Parameters
    ----------
    E : ndarray
        Point in R^2
    E_i : ndarray
        One of our data points

    Returns
    -------
    float
        Value at E of gaussian centered at E_i
    """
    return m.exp(-euclidean(E, E_i)**2)

def H(E, D):
    """
    The Hamiltonian function. Defines total energy of the system in terms
    of position and momentum. We use the notation E = (x, p) to denote
    points in phase space.

    Parameters
    ---------
    E : ndarray
        Point in R^2
    D : ndarray
        Array of data points E_1, ..., E_n in R^2

    Returns
    -------
    float
        Value of H at point E
    """

    return sum(H_i(E, E_i) for E_i in D)

def H_x(E, D):
    """
    1st partial derivative of H with respect to x.

    Parameters
    ---------
    E : ndarray
        Point in R^2
    D : ndarray
        Array of data points E_1, ..., E_n in R^2

    Returns
    -------
    float
        Value of H_x at point E
    """
    return sum(-2*(E[0]-E_i[0])*H_i(E, E_i) for E_i in D)

def H_p(E, D):
    """
    1st partial derivative of H with respect to p.

    Parameters
    ---------
    E : ndarray
        Point in R^2
    D : ndarray
        Array of data points E_1, ..., E_n in R^2

    Returns
    -------
    float
        Value of H_p at point E
    """
    return sum(-2*(E[1]-E_i[1])*H_i(E, E_i) for E_i in D)

def f_1(E, D, k1):
    """
    First modifying function for our dynamical system. Steers the
    solution from the initial point to a point on the desired trajectory S 
    in finite time.

    Parameters
    ---------
    E : ndarray
        Point in R^2
    D : ndarray
        Array of data points E_1, ..., E_n in R^2
    k1 : float
        Tuning parameter. k1 > 0

    Returns
    -------
    float
        Value of f_1 at point E
    """
    return k1/(H_x(E, D)**2 + H_p(E, D)**2)

def f_2(E, D, k2):
    """
    Second modifying function for our dynamical system. Allows the
    trajectory S to be computed in finite time.

    Parameters
    ---------
    E : ndarray
        Point in R^2
    D : ndarray
        Array of data points E_1, ..., E_n in R^2
    k1 : float
        Tuning parameter. k2 > 0

    Returns
    -------
    float
        Value of f_2 at point E
    """
    return k2/m.sqrt(H_x(E, D)**2 + H_p(E, D)**2)

def trajectory_reached(E, D, H_r, delta):
    """
    Determines if the solution of the first stage of the modified dynamics
    has reached the trajectory S to be computed, ie. if H(E) = H_r.

    Parameters
    ----------
    E : ndarray
        Point in R^2
    D : ndarray
        Array of data points E_1, ..., E_n in R^2
    H_r : float
        Reference level for the trajectory to be computed. 0 < H_r < 1
    delta : float
        Target error for H

    Returns
    -------
    bool
        True if H(E) is within delta of H_r, False otherwise
    """
    return abs(H(E, D)-H_r) < delta

def is_periodic():
    """
    Determines if the solution of the second stage of the modified dynamics
    is periodic, ie. if there exists time T such that S(t) = S(t + T) for
    all t
    """
    pass