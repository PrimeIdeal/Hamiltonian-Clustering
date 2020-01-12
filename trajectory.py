"""
We store functions for defining the trajectories of the 
Hamiltonian system in this module.

Requirements: numpy, math

Date            Author              Description
10/15/2019      Pierre Gauvreau     initial version - leapfrog algorithm
01/11/2020      Pierre Gauvreau     updates corresponding to class based ODESolve.py
"""


import numpy as np 
import math as m

def euclidean(E1, E2):
    """
    Returns Euclidian distance between two points E1 and E2 in phase space.

    Parameters
    ----------
    E1 : ndarray
        Point in phase space
    E2 : ndarray
        Point in phase space

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
        Point in phase space
    E_i : ndarray
        One of our data points

    Returns
    -------
    float
        Value at E of gaussian centered at E_i
    """

    return m.exp(-euclidean(E, E_i)**2)

def H(D, E):
    """
    The Hamiltonian function. Defines total energy of the system in terms
    of position and momentum. We use the notation E = (x, p) to denote
    points in phase space.

    Parameters
    ---------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E : ndarray
        Point in phase space

    Returns
    -------
    float
        Value of H at point E
    """

    return sum(H_i(E, E_i) for E_i in D)

def H_partial(D, E, var):
    """
    1st partial derivative of H with respect to some variable.

    Parameters
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E : ndarray
        Point in phase space
    var : str
        The variable with respect to which we take the derivative

    Returns
    -------
    float
        Value of H_var at point E
    """

    var_dict = {'x': 0, 'p': 1}

    assert var in var_dict.keys()

    idx = var_dict[var]
    return sum(-2*(E[idx]-E_i[idx])*H_i(E, E_i) for E_i in D)

def f(D, E, k, stage):
    """
    Modifying function for our dynamical system. First stage function steers 
    the solution from the initial point to a point on the desired trajectory S 
    in finite time. Second stage function allows the trajectory S to be 
    computed in finite time.

    Parameters
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E : ndarray
        Point in phase space
    k : tuple(float, float)
        Tuning parameters for the stage. k1, k2 > 0
    stage : int
        stage of the trajectory calculation we are computing, 1 or 2

    Returns
    -------
    float
        Value of f at point E
    """

    assert stage in (1, 2)

    if stage == 1:
        return k[0]/(H_partial(D, E, 'x')**2 + H_partial(D, E, 'p')**2)
    else:
        return k[1]/m.sqrt(H_partial(D, E, 'x')**2 + H_partial(D, E, 'p')**2)

def dynamics(D, E, H_r, stage, k):
    """
    The modified dynamics to be solved in the trajectory computation. Note
    that x and p have no explicit time dependence.

    Parameters
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E : ndarray
        Point in phase space
    H_r : float
        Reference level for the trajectory to be computed. Must be in (0, 1)
    stage : int
        Current stage of the trajectory computation
    k : tuple(float, float)
        Tuning parameters for the modifying function

    Returns
    -------
    float
        Value of the time derivatives at (E, t)
    """

    Hx, Hp, diff, mod = H_partial(D, E, 'x'), H_partial(D, E, 'p'), H(D, E)-H_r, f(D, E, k, stage)
    dxdt, dpdt = mod*(Hp - Hx*diff**(1/3)), -mod*(Hx + Hp*diff**(1/3))

    return np.array([dxdt, dpdt], float)