# We have a small resolution ERA5 data in ClimaArtifacts
# that we can run locally
# make it so if high resolution is not available, run with low resolution, which is available for everyone and not just hpc


# # Global bucket run

# The code sets up and runs the bucket model  on a spherical domain,
# using ERA5 data.

# Simulation Setup
# Number of spatial elements: 101 in horizontal, 5 in vertical
# Soil depth: 3.5 m
# Simulation duration: 365 d
# Timestep: 3600 s
# Timestepper: RK4
# Atmos forcing update: every 3 hours
import SciMLBase
import ClimaComms
ClimaComms.@import_required_backends
import ClimaTimeSteppers as CTS
using ClimaCore
using ClimaUtilities.ClimaArtifacts
import Interpolations
using Insolation

using ClimaDiagnostics
using ClimaAnalysis
import ClimaAnalysis.Visualize as viz
using ClimaUtilities

import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearInterpolation, PeriodicCalendar
import ClimaUtilities.SpaceVaryingInputs: SpaceVaryingInput
import ClimaUtilities.Regridders: InterpolationsRegridder
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import ClimaParams as CP

using ClimaLand
using ClimaLand.Bucket:
    BucketModel, BucketModelParameters, PrescribedBaregroundAlbedo
import ClimaLand
import ClimaLand.Parameters as LP

using Statistics
using CairoMakie
using Dates
using Random
import NCDatasets
using NCDatasets
using StaticArrays
using CUDA

import EnsembleKalmanProcesses as EKP

const FT = Float64;
context = ClimaComms.context()
device = ClimaComms.device()
device_suffix = device isa ClimaComms.CPUSingleThreaded ? "cpu" : "gpu"
root_path = "bucket_longrun"

# Returns a 2 SVectors of (lat, lon) tuples for n random locations on the land
# surface. The two sets of locations are designed to be used as a training and
# validation set.
function rand_locations(surface_space, regridder_type, n = 100)
    # Load the landsea mask data
    datapath = ClimaLand.Artifacts.topmodel_data_path()
    landsea_mask = SpaceVaryingInput(
        datapath,
        "landsea_mask",
        surface_space;
        regridder_type,
    )

    # Get the corresponding latitude and longitude values
    lat = ClimaCore.Fields.coordinate_field(surface_space).lat
    lon = ClimaCore.Fields.coordinate_field(surface_space).long

    # Find the coordinates of 2n random land locations
    land_inds = rand(findall(x -> x == 1.0, Array(parent(landsea_mask))), 2 * n)

    # Since this is run very rarely (once at start of calibration run), we don't
    # mind scalar iteration when running on the GPU
    CUDA.@allowscalar land_locs = StaticArrays.sacollect(
        SVector{2 * n, Tuple{FT, FT}},
        zip(parent(lon)[land_inds], parent(lat)[land_inds]),
    )

    # Return a uniform random sample of n land locations
    return (
        SVector{n}(land_locs[1:n]...),
        SVector{n}(land_locs[(n + 1):end]...),
    )
end

"""
    era5_monthly_stdevs(varname::String)

Returns the year over year standard deviation of a given variable from ERA5.
i.e, a 12 element vector of the standard deviation of the variable for each
month taken across all data years. Assumes you're looking for a variable
contained in the era5_monthly_averages_surface_single_level_197901-202410.nc
data file.
"""
function era5_monthly_stdevs(varname::String, lat, lon)
    # Read the data for this variable from all data years
    era5_ds = Dataset(
        joinpath(
            ClimaLand.Artifacts.era5_surface_data_1979_2024_path(),
            "era5_monthly_averages_surface_single_level_197901-202410.nc",
        ),
    )

    # Create a static array to store the stddevs in
    stdevs = MVector{12, FT}(undef)

    # Find the nearest latitude and longitude to the target location in the data
    lat_ind = argmin(abs.(era5_ds["latitude"][:] .- lat))
    lon_ind = argmin(abs.(era5_ds["longitude"][:] .- lon))

    for i in 1:12
        monthly_data = era5_ds[varname][
            lon_ind,
            lat_ind,
            findall((x) -> month(x) == i, era5_ds["time"]),
        ]
        stdevs[i] = std(monthly_data)
    end

    return stdevs
end


# function target_and_locations()
# Define the locations longitudes and latitudes
# Needs to be defined once, both for g(θ) and ERA5 target
t0 = 0.0
tf = 60 * 60.0 * 24 * 366
Δt = 900.0
nelements = (101, 7)
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
locations, validation_locations =
    rand_locations(surface_space, regridder_type, 25)

#     (; κ_soil, ρc_soil, f_bucket, W_f, p, z_0m) = params
# The truth params = (;κ_soil = FT(1.5), ρc_soil = FT(2e6), f_bucket = FT(0.75), W_f = FT(0.2), p = FT(1), z_0m = FT(1e-2))
# Read in the era5 datafile
era5_ds = Dataset(
    joinpath(
        ClimaLand.Artifacts.era5_surface_data2008_path(),
        "era5_monthly_surface_fluxes_200801-200812.nc",
    ),
)

# Make the ERA5 target
ERA5_target = []
close_location = (x, y) -> abs(x - y) < 5e-1
for (lon, lat) in locations
    # Fetch slices of lhf and shf era5 data from the era5 dataset
    lat_ind, lon_ind =
        findall((x) -> close_location(x, lat), era5_ds["latitude"][:])[1],
        findall((x) -> close_location(x, lon + 180), era5_ds["longitude"][:])[1]
    lhf_loc = vec(era5_ds["mslhf"][lon_ind, lat_ind, :][:, end, :])
    shf_loc = vec(era5_ds["msshf"][lon_ind, lat_ind, :][:, end, :])
    lhf_ERA5_std = era5_monthly_stdevs("mslhf", lat, lon)
    shf_ERA5_std = era5_monthly_stdevs("msshf", lat, lon)

    # Create Observation objects for lhf and shf
    lhf_ERA5 = EKP.Observation(
        Dict(
            "samples" => lhf_loc,
            "covariances" => cov(lhf_loc) * EKP.I,
            "names" => "lhf_$(lon)_$(lat)",
        ),
    )

    shf_ERA5 = EKP.Observation(
        Dict(
            "samples" => shf_loc,
            "covariances" => cov(shf_loc) * EKP.I,
            "names" => "shf_$(lon)_$(lat)",
        ),
    )

    # Add the observations to the target
    push!(ERA5_target, lhf_ERA5)
    push!(ERA5_target, shf_ERA5)

end

full_obs_era5 = EKP.combine_observations(ERA5_target)
observations = EKP.get_obs(full_obs_era5)
#    return training_locations, observations
# end


# Things to consider:
# - masking out the ocean when comparing to data
# - output paths matching to parameters

# Priors:
# κ_soil = FT(1.5): Gaussian with mean = 2, std dev = 1, cannot be negative [ or uniform from 0.5 to 2.5]
# ρc_soil = FT(2e6); Gaussian with mean = 4e6, std dev = 2e6, cannot be negative [ or uniform from 1e5 to 5e6]
# f_bucket = FT(0.75); Gaussian with mean = 0.5, std_dev = 0.3, cannot be negative, cannot be greater than 1 [ or uniform from 0.2 to 0.9]
# W_f = FT(0.2); Gaussian with mean = 0.4, std_dev = 0.4, cannot be negative [ or uniform from 0.05 to 1]
# p = FT(1); we can range uniform 1 to 2.5, cannot be smaller than 1
#z_0m = FT(1e-2); can we sample from log normal ranging from z_0m = 1e-3 to z_0m = 0.1?
