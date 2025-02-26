# Define the locations longitudes and latitudes
# Needs to be defined once, both for g(θ) and ERA5 target
t0 = 0.0
tf = 60 * 60.0 * 24 * 366
Δt = 900.0
nelements = (50, 5)
regridder_type = :InterpolationsRegridder
radius = FT(6378.1e3)
depth = FT(3.5)
domain = ClimaLand.Domains.SphericalShell(;
    radius = radius,
    depth = depth,
    nelements = nelements,
    npolynomial = 1,
    dz_tuple = FT.((1.0, 0.05)),
)
surface_space = domain.space.surface
training_locations, validation_locations =
    rand_locations(surface_space, regridder_type, 100)
