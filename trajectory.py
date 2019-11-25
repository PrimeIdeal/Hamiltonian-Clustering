"""
We store functions for defining and solving the trajectories of the 
Hamiltonian system in this module.

General approach: given data set D = {E_1, E_2, ..., E_n} in phase space where
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

def H(E):
    """
    The Hamiltonian function. Defines total energy of the system in terms
    of position and momentum. We use the notation E = (x, p) to denote
    points in phase space.

    Parameters
    ---------
    E : ndarray
        Point in phase space

    Returns
    -------
    float
        Value of H at point E
    """

    return sum(H_i(E, E_i) for E_i in D)

def H_partial(E, var):
    """
    1st partial derivative of H with respect to some variable.

    Parameters
    ----------
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

def f(E, k, stage):
    """
    Modifying function for our dynamical system. First stage function steers 
    the solution from the initial point to a point on the desired trajectory S 
    in finite time. Second stage function allows the trajectory S to be 
    computed in finite time.

    Parameters
    ----------
    E : ndarray
        Point in phase space
    k : float
        Tuning parameter for the stage. k > 0
    stage : int
        stage of the trajectory calculation we are computing, 1 or 2

    Returns
    -------
    float
        Value of f at point E
    """

    assert stage in (1, 2)

    if stage == 1:
        return k/(H_partial(E, 'x')**2 + H_partial(E, 'p')**2)
    else:
        return k/m.sqrt(H_partial(E, 'x')**2 + H_partial(E, 'p')**2)

def dynamics(E, t):
    """
    The modified dynamics to be solved in the trajectory computation. Note
    that x and p have no explicit time dependence.

    Parameters
    ----------
    E : ndarray
        Point in phase space
    t : float
        Current time

    Returns
    -------
    float
        Value of the time derivatives at (E, t)
    """

    Hx, Hp, diff, mod = H_partial(E, 'x'), H_partial(E, 'p'), H(E)-H_r, f(E, k, stage)
    dxdt, dpdt = mod*(Hp - Hx*diff**(1/3)), -mod*(Hx + Hp*diff**(1/3))

    return np.array([dxdt, dpdt], float)

def compute_trajectory(E_initial, data, H_r, h, k1, k2, delta1, delta2, maxcount=5):
    """
    Computes trajectory corresponding to E_initial and reference level ref_level, subject
    to the Hamiltonian defined by the points in dataset.

    Parameters
    ----------
    E_initial : ndarray
        Initial point of the trajectory in phase space
    data : array
        The array of our observed data points
    ref_level : float
        The reference level for the trajectory to be computed. Must be in (0, 1)
    h : float
        Step size for the ODE solver. Must be > 0
    k1 : float
        Tuning parameter for stage 1 of the trajectory computation. Must be > 0
    k2 : float
        Tuning parameter for stage 2 of the trajectory computation. Must be > 0
    delta1 : float
        Target error for stage 1 of the trajectory computation. Must be < h
    delta2 : float
        Target error for stage 2 of the trajectory computation. Must be < h
    maxcount : int
        Number of consecutive checks on periodicity before trajectory is 
        determined to be closed. Defaults to 5

    Returns
    -------
    list
        x-coordinates of the trajectory S
    list
        p-coordinates of the trajectory S
    """

    def trajectory_reached(E, H_r, delta1):
        """
        Determines if the solution of the first stage of the modified dynamics
        has reached the trajectory S to be computed, ie. if H(E) = H_r.

        Parameters
        ----------
        E : ndarray
            Point in phase space
        H_r : float
            Reference level for the trajectory to be computed. 0 < H_r < 1
        delta : float
            Target error for H

        Returns
        -------
        bool
            True if H(E) is within delta of H_r, False otherwise
        """
        
        return abs(H(E)-H_r) < delta1

    D, H_r = data, ref_level
    S = E_initial
    t1, t2 = 0, 0

    # Stage 1
    stage = 1
    k = k1
    half = None 

    # ADDITIONAL FEATURES
    # -------------------
    # add visualization/analysis for optimizing convergeance
    # output E_current at each step
    
    # add convergence counter
    iter_counter = 0
    iter_max = 1000

    # conditional 1 on reaching target accuracy of delta1 w.r.t. |H(E) - H_r|
    # conditional 2 on reaching maximum of 1000 iterations
    while not trajectory_reached(S, H_r, delta1) and iter_counter <= iter_max:
        S, half = leapfrog(S, half, dynamics, t1, h)
        t1 += h
        iter_counter = iter_counter + 1

    # Stage 2
    stage = 2
    k = k2
    closed = False
    count_within_range = 0
    half = None
    x_list, p_list = [S[0]], [S[1]] # need to replace with something faster than lists

    while not closed:
        S, half = leapfrog(S, half, dynamics, t2, h)
        x_list.append(S[0])
        p_list.append(S[1])
        t2 += h
        S_inspected = np.array([x_list[count_within_range], p_list[count_within_range]])
        if euclidean(S_inspected, S) < delta2:
            count_within_range += 1
        else:
            if count_within_range > 0:
                count_within_range = 0
        if count_within_range >= maxcount:
            closed = True
    
    return x_list[:-maxcount], p_list[:-maxcount]