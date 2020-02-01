"""
We store functions for solving systems of ordinary differential equations
in this module. 

Notation: functions solve 1st order ODE systems in the form
    dx/dt = f(x, t)
where
    x = (x1, x2, ..., xn)
    f(x, t) = (f_x1(x, t), f_x2(x, t), ..., f_xn(x, t))

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

Guidelines:
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
import trajectory as tr


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
        Reference level for the trajectory to be computed. Must be in (0, 1)
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

    def __init__(self, D, H_r, delta, **kwargs):
        self._D = D
        self._E_curr = None
        self._H_r = H_r
        self._delta = delta
        self._x_list = []
        self._p_list = []
        self._num_steps = 0
        self._closed_count = 0

    def update_trajectory(self):
        """
        Adds E_curr's coordinates to x_list and p_list
        """
        self._x_list.append(self._E_curr[0])
        self._p_list.append(self._E_curr[1])

    def trajectory_reached(self):
        """
        Determines if the solution of the first stage of the modified dynamics 
        has reached the trajectory S to be computed, ie. if H(E) = H_r

        Returns
        -------
        bool
            True if H(E) is within delta of H_r, False otherwise
        """

        return abs(tr.H(self._D, self._E_curr)-self._H_r) < self._delta[0]

    def trajectory_closed(self, min_count):
        """
        Determines if the solution of the second stage of the modified dynamics
        has achieved a closed trajectory, ie. if there exists T such that 
        S(t) = S(t+T) for all t

        Parameters
        ----------
        min_count : int
            Minimum number of consecutive trajectory points S[i+k] within specified
            distance of points S[i] required to declare S closed

        Returns
        -------
        bool
            True if S is closed, False otherwise
        """

        ref_pt = np.array([self._x_list[self._closed_count], self._p_list[self._closed_count]])

        if tr.euclidean(ref_pt, self._E_curr) < self._delta[1]:
            self._closed_count += 1
            if self._closed_count == min_count:
                return True
        else:
            if self._closed_count > 0:
                self._closed_count = 0
        return False

    @property
    def num_steps(self):
        """
        Returns the current number of iterations used during the trajectory
        calculation

        Returns
        -------
        int
            Current number of steps
        """

        return self._num_steps

    @abstractmethod
    def solve(self, E_initial, h, k, min_count):
        pass

    def clear_comp(self):
        """Clears all instance variables modified by solve()"""
        self._E_curr = None
        self._x_list = []
        self._p_list = []
        self._num_steps = 0
        self._closed_count = 0


class Leapfrog(ODESolver):
    """
    Solves both stages of the dynamical system using the leapfrog algorithm.
    Time-reversal symmetry is guaranteed

    Attributes
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E_curr : ndarray
        A point in phase space. The most recently computed point of
        the solution
    H_r : float
        Reference level for the trajectory to be computed. Must be in (0, 1)
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

    def __init__(self, D, H_r, delta, **kwargs):
        ODESolver.__init__(self, D, H_r, delta)

    def _euler(self, h, k, stage):
        """
        Helper function for Leapfrog.solve(). 

        Executes a single iteration of Euler's method with time step h/2: Given 
        x(t), returns x(t+h/2).
        """

        return self._E_curr + 0.5*h*tr.dynamics(self._D, self._E_curr,
                                                self._H_r, stage, k)
    
    def solve(self, E_initial, h, k, min_count=5):
        """
        Implements leapfrog algorithm to solve dynamical system

        Parameters
        ----------
        E_initial : ndarray
            A point in phase space. Initial condition of the dynamical
            system
        h : float
            The step size used by the algorithm
        k : tuple(float, float)
            Tuning parameters for the modifying function of the dynamical
            system
        min_count : int
            Number of consecutive checks on periodicity before trajectory
            is determined to be closed. Defaults to 5

        Returns
        -------
        list(float)
            The ordered x-coordinates of the solution trajectory S
        list(float)
            The ordered p-coordinates of the solution trajectory S
        """

        self._E_curr = E_initial
        
        # Stage 1
        half = self._euler(h, k, 1)
        while not self.trajectory_reached():
            self._E_curr += h*tr.dynamics(self._D, half, self._H_r, 1, k)
            half += h*tr.dynamics(self._D, self._E_curr, self._H_r, 1, k)

        # Stage 2
        self.update_trajectory()

        half = self._euler(h, k, 2)
        while not self.trajectory_closed(min_count):
            self._E_curr += h*tr.dynamics(self._D, half, self._H_r, 2, k)
            half += h*tr.dynamics(self._D, self._E_curr, self._H_r, 2, k)
            self.update_trajectory()
        return self._x_list[-min_count:], self._p_list[-min_count:]


class AdaptiveRK4(ODESolver):
    """
    Solves both stages of the dynamical system using Adaptive RK4.
    Time-reversal symmetry is not guaranteed

    Attributes
    ----------
    D : set(ndarray)
        The data set containing our observed points in phase space
    E_curr : ndarray
        A point in phase space. The most recently computed point of
        the solution
    H_r : float
        Reference level for the trajectory to be computed. Must be in (0, 1)
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
    step_delta : float
        Target accuracy parameter for the adaptive step
    """

    def __init__(self, D, H_r, delta, **kwargs):
        super().__init__(D, H_r, delta)
        self._step_delta = kwargs.get('step_delta', None)

        assert self._step_delta, 'A nonzero step_delta must be passed in.'

    def _rho(self, E1, E2, step):
        """
        Helper function for AdaptiveRK4._adaptive_step().

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

        Returns
        -------
        float
            Ratio of actual to target accuracy
        """

        diff = E1 - E2
        return 30*step*self._step_delta/m.sqrt(np.dot(diff, diff))

    def _fixed_step(self, E, stage, k, step):
        """
        Helper function for AdaptiveRK4.solve().

        A single iteration of the RK4 algorithm with fixed step size.

        Parameters
        ----------
        E : ndarray
            A point in phase space
        stage : int
            The current stage of the trajectory computation
        k : tuple(float, float)
            Tuning parameters for the modifying function
        step : float
            A new step size passed in by AdaptiveRK4
        """

        k1 = step*tr.dynamics(self._D, E, self._H_r, stage, k)
        k2 = step*tr.dynamics(self._D, E + 0.5*k1, self._H_r, stage, k)
        k3 = step*tr.dynamics(self._D, E + 0.5*k2, self._H_r, stage, k)
        k4 = step*tr.dynamics(self._D, E + k3, self._H_r, stage, k)

        return E + (k1+2*k2+2*k3+k4)/6

    def solve(self, E_initial, h, k, min_count=5):
        """
        Implements Adaptive RK4 to solve dynamical system

        Parameters
        ----------
        E_initial : ndarray
            A point in phase space. Initial condition of the dynamical
            system
        h : float
            The step size used by the algorithm
        k : tuple(float, float)
            Tuning parameters for the modifying function of the dynamical
            system
        min_count : int
            Number of consecutive checks on periodicity before trajectory
            is determined to be closed. Defaults to 5

        Returns
        -------
        list(float)
            The ordered x-coordinates of the solution trajectory S
        list(float)
            The ordered p-coordinates of the solution trajectory S
        """

        self._E_curr = E_initial

        def update_step(stage, step_num):
            """Returns E1 and E2 at the appropriate stage,"""
            e1 = self._fixed_step(
                self._fixed_step(self._E_curr, stage, k, step_num),
                stage, k, step_num)
            e2 = self._fixed_step(self._E_curr, stage, k, 2 * step_num)
            return e1, e2

        # Stage 1
        while not self.trajectory_reached():
            ratio = 0
            step = h

            while ratio < 1:
                E1, E2 = update_step(1, step)
                ratio = self._rho(E1, E2, step)
                step *= ratio**(1/4)

            self._E_curr = self._fixed_step(self._E_curr, 1, k, step)

        # Stage 2
        self.update_trajectory()

        while not self.trajectory_closed(min_count):
            ratio = 0
            step = h

            while ratio < 1:
                E1, E2 = update_step(2, step)
                ratio = self._rho(E1, E2, step)
                step *= ratio ** (1 / 4)

            self._E_curr = self._fixed_step(self._E_curr, 2, k, step)
            self.update_trajectory()

        return self._x_list[: -min_count], self._p_list[: -min_count]
