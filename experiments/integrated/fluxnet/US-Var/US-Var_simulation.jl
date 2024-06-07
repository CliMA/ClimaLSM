"""This file contains simulation variables for running Clima Land on the US-Var
fluxtower site. This includes both the domain variables and timestepping 
variables for running the simulation."""

# DOMAIN SETUP:

# Column dimensions - separation of layers at the top and bottom of the column:
dz_bottom = FT(1.0)
dz_top = FT(0.05)

# Stem and leaf compartments and their heights:
h_leaf = FT(0.7) # m
h_stem = FT(0) # m

# TIME STEPPING:

# Starting time:
t0 = Float64(21 * 3600 * 24)# start day 21 of the year

# Time step size:
dt = Float64(40)

# Number of timesteps between saving output:
n = 45
