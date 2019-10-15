"""
We solve the trajectories for our Hamiltonian system in this module.

General approach: given data set D = {E_1, E_2, ..., E_n} in R^2 where
E_i = (x_i, p_i), define the Hamiltonian function
    H(E) = sum(H_i(E)) for 1 < i < N
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
until we have computed a closed trajectory.

Requirements: numpy, math, ODESolve

Date            Author              Description
10/15/2019      Pierre Gauvreau     initial version - leapfrog algorithm
"""

import numpy as np 
import math as m
from ODESolve import leapfrog

