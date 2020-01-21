"""
We store functions for solving systems of ordinary differential equations
in this module. 

Notation: functions solve 1st order ODE systems in the form
    dx/dt = f(x, t)
where
    x = (x1, x2, ..., xn)
    f(x, t) = (f_x1(x, t), f_x2(x, t), ..., f_xn(x, t))

Guidelines:
    - for now, the DE solvers should execute a single step of the solution
    per function call
    - solvers should accept a time argument even if the solution has no
    explicit time dependence
    - for now, solvers should be implementations of time-reversal symmetric 
    algorithms (although we might be able to relax this restriction)

Requirements: numpy, math

Date            Author              Description
10/14/2019      Pierre Gauvreau     Initial version: leapfrog algorithm
11/19/2019      Pierre Gauvreau     Adaptive RK4
"""

import numpy as np 
import math as m 

# Leapfrog Algorithm

def euler(E, f, t, h):
    """
    Helper function for leapfrog. Executes a single iteration of Euler's 
    method with time step h/2. Given x(t), returns x(t+h/2).

    Parameters
    ----------
    E : ndarray
        The solution at time t
    f : function
        The system to be solved
    t : float
        Current time
    h : float
        Step size for leapfrog algorithm

    Returns
    -------
    ndarray
        The solution at time t+h/2
    """
    return E + 0.5*h*f(E, t)

def leapfrog(E, half, f, t, h):
    """
    Executes a single iteration of the leapfrog algorithm with time step h.
    Given x(t) and x(t+h/2), returns x(t+h) and x(t+3h/2).

    Parameters
    ----------
    E : ndarray
        The solution at time t
    half : ndarray
        The solution at time t+h/2
    f : function
        The system to be solved
    t : float
        Current time
    h : float
        Step size for leapfrog algorithm

    Returns
    -------
    ndarray
        The solution at time t+h
    ndarray
        The solution at time t+3h/2
    """
    if half is None:
        half = euler(E, f, t, h)
    new_E = E + h*f(half, t + 0.5*h)
    new_half = half + h*f(new_E, t + h)

    return new_E, new_half

# Adaptive RK4

def rho(E1, E2, step, delta):
    """
    Helper function for adaptive RK4.

    Computes the ratio of actual to target accuracy for a given
    step size h, target accuracy delta, and two estimates E1, E2
    of E(t+2h).

    Parameters
    ----------
    E1 : ndarray
        A point in phase space
    E2 : ndarray
        A point in phase space
    step : float
        Designated step size
    delta : float
        Designated target accuracy

    Returns
    -------
    float
        Ratio of actual to target accuracy
    """
    diff = E1 - E2
    return 30*step*delta/m.sqrt(np.dot(diff, diff))

def FixedRK4(E, f, t, step):
    """
    Helper function for AdaptiveRK4.

    A single iteration of the RK4 algorithm with fixed step size.

    Parameters
    ----------
    E : ndarray
        A point in phase space
    f : function
        The ODE system to be solved
    t : float
        current time
    step : float
        a new step size passed in by AdaptiveRK4
    """

    k1 = step*f(E, t)
    k2 = step*f(E + 0.5*k1, t + 0.5*step)
    k3 = step*f(E + 0.5*k2, t + 0.5*step)
    k4 = step*f(E + k3, t + step)

    return E + (k1+2*k2+2*k3+k4)/6

def AdaptiveRK4(E, f, t, h, delta):
    """
    A single iteration of Adaptive RK4.

    Parameters
    ----------
    E : ndarray
        A point in phase space
    f : function
        The ODE system to be solved
    t : float
        Current time
    h : float
        Initial step size
    delta : float
        Target accuracy

    Returns
    -------
    ndarray
        A point in phase space
    float
        The new step size
    """

    ratio = 0
    step = h 

    while ratio < 1:
        E1 = FixedRK4(FixedRK4(E, f, t, step), f, t+step, step)
        E2 = FixedRK4(E, f, t, 2*step)
        ratio = rho(E1, E2, step, delta)
        step = step*ratio**(1/4)

    return FixedRK4(E, f, t, step), step