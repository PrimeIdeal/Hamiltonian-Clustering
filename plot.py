"""A collection of methods for executing the Hamiltonian Clustering algorithm"""
import ODESolve as ode_solve
import numpy as np
from load_create import create_test_set
from integrals import monte_mean_value as winding_number
import matplotlib.pyplot as plt
from scipy import interpolate

# TODO: Get rid of this and think of something more clever
PERIODICITY_STEPS = 5
TARGET_DELTA = (0.1, 0.1)
STEP_DELTA = 0.2
INTEGRAL_ERROR = 1 / 2
INTEGRAL_NUM_SAMPLES = 50


class ClusterPoints:
    """
    A class to fully encapsulate the Hamiltonian Clustering algorithm along with
    plotting capabilities.
    """

    def __init__(self, x_range, p_range, num_pts, num_groups, H_r,
                 step_size=0.1, k_params=(1, 1),
                 ode_solver=ode_solve.AdaptiveRK4):
        """Creates an instance holding all problem context.

        Parameters
        ----------
        x_range : ndarray
            The x boundaries of the rectangle
        p_range : ndarray
            The p boundaries of the rectangle
        num_pts : int
            The total number of points to be generated
        num_groups : int
            The total number of test clusters to be generated
        H_r : float
            Reference level for the trajectory to be computed. Must be in (0, 1)
        step_size : float
            The step size used by the ODE solvers
        k_params : tuple(float, float)
            Tuning parameters for the modifying function of the dynamical
            system
        ode_solver : Class(ODESolver)
            The ODESolver class to use for our computations.
        """

        self.x_range = x_range
        self.p_range = p_range
        self.num_pts = num_pts
        self.num_groups = num_groups
        self._hr = H_r
        self.step_size = step_size
        self.k_params = k_params
        self.data_set = create_test_set(x_range=x_range, p_range=p_range,
                                        num_pts=num_pts, num_groups=num_groups)
        self.ode_solver = ode_solver
        self.level_sets = []

    def cluster_points(self):
        """Run the point clustering algorithm."""
        unclustered_points = self.data_set.copy()
        solver = self.ode_solver(D=unclustered_points, H_r=self._hr,
                                 delta=TARGET_DELTA, step_delta=STEP_DELTA)

        while unclustered_points:
            e_i = unclustered_points.pop()
            cur_cluster = set()

            # TODO: I think there's a conceptual error in my understanding of
            #  monte_mean_value, why should it take random_gen instead of
            #  sol_trajectory
            sol_trajectory = solver.solve(E_initial=e_i, h=self.step_size,
                                          k=self.k_params,
                                          min_count=PERIODICITY_STEPS)
            self.level_sets.append(sol_trajectory)
            for e_k in unclustered_points:
                if abs(1 - winding_number(
                        D=self.data_set, E_i=e_k, H_r=self._hr,
                        x_range=self.x_range, p_range=self.p_range,
                        n=INTEGRAL_NUM_SAMPLES)) < INTEGRAL_ERROR:
                    cur_cluster.add(e_k)
            unclustered_points -= cur_cluster
            self.ode_solver.clear_comp()

    def plot(self):
        """Visually plots both the data and each locus."""

        plt.figure()

        # Plot the data points
        x_points, p_points = zip(*self.data_set)
        plt.plot(x_points, p_points, '.')

        # Plot the level sets
        for level_set in self.level_sets:
            tck, u = interpolate.splprep(level_set, s=0.001)
            out = interpolate.splev(u, tck)
            plt.plot(*out)
        plt.show()


# Testing
if __name__ == '__main__':
    test_run = ClusterPoints((0, 2), (0, 2), 10, 5, 0.4)
    test_run.cluster_points()
    test_run.plot()
