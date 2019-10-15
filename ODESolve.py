"""
We store functions for solving systems of ordinary differential equations
in this module. 

Notation: functions solve 1st order ODE systems in the form
    dx/dt = f(x, t)
where
    x = (x1, x2, ..., xn)
    f(x, t) = (f_x1(x, t), f_x2(x, t), ..., f_xn(x, t))

Guidelines:
    - for now, the DE solvers should execute a single step of the solution(s)
    per function call
    - solvers should accept a time argument even if the solution has no
    explicit time dependence
    - solvers must be implementations of time-reversal symmetric algorithms

Requirements: numpy, math

Date            Author              Description
10/14/2019      Pierre Gauvreau     Initial version: leapfrog algorithm
"""

import numpy as np 
import math as m 

def euler(x, f, t, h):
    """
    Helper function for leapfrog. Executes a single iteration of Euler's 
    method with time step h/2: given x(t), returns x(t+h/2).

    Parameters
    ----------
    x : ndarray
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
    return x + 0.5*h*f(x, t)

def leapfrog(x, half, f, t, h):
    """
    Executes a single iteration of the leapfrog algorithm with time step h:
    given x(t) and x(t+h/2), returns x(t+h) and x(t+3h/2).

    Parameters
    ----------
    x : ndarray
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
    new_x = x + h*f(half, t + 0.5*h)
    new_half = half + h*f(new_x, t + h)

    return new_x, new_half