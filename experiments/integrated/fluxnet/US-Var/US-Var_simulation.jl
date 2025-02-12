"""This file contains simulation variables for running Clima Land on the US-Var
fluxtower site. This includes both the domain variables and timestepping
variables for running the simulation."""

# DOMAIN SETUP:

# Column dimensions
# Stem and leaf compartments and their heights:
n_stem = Int64(0)
n_leaf = Int64(1)
h_leaf = FT(0.5) # m, Xu and Baldocchi, 2003
h_stem = FT(0) # m
h_canopy = h_stem + h_leaf
# TIME STEPPING:

# Starting time:
t0 = Float64(0)

# Time step size:
dt = Float64(900)
# For soil column
nelements = 14
zmin = FT(-0.5) #m, Xu and Baldocchi, 2003
zmax = FT(0)
dz_tuple = nothing
