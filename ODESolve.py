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

Requirements: numpy, math, trajectory, abc

Date            Author              Description
10/14/2019      Pierre Gauvreau     Initial version: leapfrog algorithm
11/19/2019      Pierre Gauvreau     Adaptive RK4
01/11/2020      Pierre Gauvreau     class based structure
"""

from abc import ABC, abstractmethod
import numpy as np 
import math as m 
from trajectory import dynamics, euclidean

# Helper functions

def euler(E, f, t, h):
    """
    Helper function for Leapfrog.update_E(). 

    Executes a single iteration of Euler's method with time step h/2: Given 
    x(t), returns x(t+h/2).

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

def rho(E1, E2, step, delta):
    """
    Helper function for AdaptiveRK4.update_E().

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
    Helper function for AdaptiveRK4.update_E().

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

# Solver classes

class ODESolver(ABC):
    """
    Abstract parent class for our ODE solvers.

    Child classes solve both stages of the dynamical system defined by
    the Hamiltonian H and the modifying function f.

    Attributes
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E_curr : ndarray
        A point in phase space. The most recently computed point of
        the solution
    H_r : float
        Reference level for the trajectory to be computed. 0 < H_r < 1
    step_size : float
        The step size used to compute the solution
    k : tuple(float)
        Tuning parameters for the modifying function f
    delta : tuple(float)
        Target accuracy parameters for stages 1 and 2 of the solution
    x_list : list
        The ordered x-coordinates of the solution
    p_list : list
        The ordered p-coordinates of the solution
    num_steps : int
        The current number of steps required to compute the trajectory
        of the dynamical system
    closed_count : int
        The current number of consecutive trajectory points S[i+k] within
        required distance of points S[i]
    """

    def __init__(self, D, E_initial, H_r, step_size, k, delta):
        self.D = D
        self.E_curr = E_initial
        self.H_r = H_r
        self.step_size = step_size
        self.k = k
        self.delta = delta
        self.x_list = []
        self.p_list = []
        self.num_steps = 0
        self.closed_count = 0

    def trajectory_reached(self):
        """
        Determines if the solution of the first stage of the modified dynamics 
        has reached the trajectory S to be computed, ie. if H(E) = H_r

        Returns
        -------
        bool
            True if H(E) is within delta of H_r, False otherwise
        """
        return abs(H(self.E_curr)-self.H_r) < self.delta[0]

    def is_closed(self, min_count):
        """
        Determines if the solution of the second stage of the modified dynamics
        has achieved a closed trajectory, ie. if there exists T such that 
        S(t) = S(t+T) for all t

        Parameters
        ----------
        min_count : int
            Minimum number of consecutive trajectory points S[i+k] within required
            distance of points S[i] to declare S closed

        Returns
        -------
        bool
            True if S is closed, False otherwise
        """
        ref_pt = np.array([self.x_list[self.closed_count], self.p_list[self.closed_count]])
        if euclidean(ref_pt, self.E_curr) < self.delta[1]:
            self.closed_count += 1
            if self.closed_count == min_count:
                return True
        else:
            if self.closed_count > 0:
                self.closed_count = 0
        return False

    def get_steps(self):
        """
        Returns the current number of iterations used during the trajectory
        calculation

        Returns
        -------
        int
            Current number of steps
        """
        return self.num_steps
            

    @abstractmethod
    def solve():
        pass


class Leapfrog(ODESolver):
    """
    """

    def __init__(self):
        ODESolver.__init__(self)
    
    def update_E(self):
        pass

class AdaptiveRK4(ODESolver):
    """
    """

    def __init__(self):
        ODESolver.__init__(self)

    def update_E(self):
        pass


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