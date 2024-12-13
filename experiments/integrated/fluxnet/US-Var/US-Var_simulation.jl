"""This file contains simulation variables for running Clima Land on the US-Var
fluxtower site. This includes both the domain variables and timestepping
variables for running the simulation."""

# DOMAIN SETUP:

# Column dimensions - separation of layers at the top and bottom of the column:
dz_bottom = FT(1.0)
dz_top = FT(0.05)

# Stem and leaf compartments and their heights:
n_stem = Int64(0)
n_leaf = Int64(1)
h_leaf = FT(0.7) # m
h_stem = FT(0) # m

# TIME STEPPING:

# Starting time:
t0 = Float64(0)

# Time step size:
dt = Float64(1800)
