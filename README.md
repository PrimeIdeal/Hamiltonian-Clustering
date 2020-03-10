# Hamiltonian-Clustering

### Authors: Pierre Gauvreau, Keirn Munro

An implementation of the algorithm proposed by Daniele Casagrande and Alessandro 
Astolfi in their 2008 paper "A Hamiltonian-Based Algorithm for Measurements 
Clustering". The clustering function is regarded as a Hamiltonian function, 
and the level lines are determined as trajectories of the associated system.

Stage 1: Basic implementation
- simple numerical methods to solve trajectories and determine cluster
validity
- manual tuning of parameters

Future developments:
- automate tuning
- more sophisticated methods for solving systems of differential
equations, numerical integration
- adapt our implementation to match the refinements in Casagrande, Astolfi and 
Sassano's 2012 paper on Hamiltonian clustering
