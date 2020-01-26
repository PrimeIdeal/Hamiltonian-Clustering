"""A collection of methods for executing the Hamiltonian Clustering algorithm"""
import ODESolve as ode_solve


class ClusterPoints:
    """
    A class to fully encapsulate the Hamiltonian Clustering algorithm along with
    plotting capabilities.
    """

    def __init__(self, x_range, p_range):
        """"""