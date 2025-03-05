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
root_path = "bucket_longrun_$(device_suffix)"

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

function setup_prob(t0, tf, Δt, params, outdir; nelements = (101, 7))
    time_interpolation_method = LinearInterpolation(PeriodicCalendar())
    regridder_type = :InterpolationsRegridder
    earth_param_set = LP.LandParameters(FT)
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
    subsurface_space = domain.space.subsurface

    start_date = DateTime(2008)
    # Forcing data
    # Forcing data
    era5_artifact_path =
        ClimaLand.Artifacts.era5_land_forcing_data2008_folder_path(; context)
    era5_ncdata_path = joinpath(era5_artifact_path, "era5_2008_1.0x1.0.nc")
    atmos, radiation = ClimaLand.prescribed_forcing_era5(
        era5_ncdata_path,
        surface_space,
        start_date,
        earth_param_set,
        FT;
        time_interpolation_method = time_interpolation_method,
        regridder_type = regridder_type,
    )

    # Set up parameters
    (; κ_soil, ρc_soil, f_bucket, W_f, p, z_0m) = params
    z_0b = z_0m
    τc = FT(Δt)
    α_snow = FT(0.8)
    albedo = PrescribedBaregroundAlbedo{FT}(α_snow, surface_space)
    bucket_parameters = BucketModelParameters(
        FT;
        albedo,
        z_0m,
        z_0b,
        τc,
        f_bucket,
        p,
        W_f,
        κ_soil,
        ρc_soil,
    )
    bucket = BucketModel(
        parameters = bucket_parameters,
        domain = domain,
        atmosphere = atmos,
        radiation = radiation,
    )

    temp_anomaly_amip(coord) = 40 * cosd(coord.lat)^4
    Y, p, cds = initialize(bucket)
    # Set temperature IC including anomaly, based on atmospheric setup
    T_sfc_0 = FT(271.0)
    @. Y.bucket.T = T_sfc_0 + temp_anomaly_amip(cds.subsurface)
    Y.bucket.W .= FT(f_bucket * W_f)
    Y.bucket.Ws .= FT(0.0)
    Y.bucket.σS .= FT(0.0)

    set_initial_cache! = make_set_initial_cache(bucket)
    set_initial_cache!(p, Y, t0)
    exp_tendency! = make_exp_tendency(bucket)

    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction(T_exp! = exp_tendency!, dss! = ClimaLand.dss!),
        Y,
        (t0, tf),
        p,
    )

    updateat = Array(t0:(3600 * 3):tf)
    drivers = ClimaLand.get_drivers(bucket)
    updatefunc = ClimaLand.make_update_drivers(drivers)

    # ClimaDiagnostics

    nc_writer = ClimaDiagnostics.Writers.NetCDFWriter(subsurface_space, outdir)

    diags = ClimaLand.default_diagnostics(
        bucket,
        start_date;
        output_writer = nc_writer,
        average_period = :monthly,
    )

    diagnostic_handler =
        ClimaDiagnostics.DiagnosticsHandler(diags, Y, p, t0; dt = Δt)

    diag_cb = ClimaDiagnostics.DiagnosticsCallback(diagnostic_handler)

    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    return prob, SciMLBase.CallbackSet(driver_cb, diag_cb)
end

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
training_locations, validation_locations =
    rand_locations(surface_space, regridder_type, 25)

#     (; κ_soil, ρc_soil, f_bucket, W_f, p, z_0m) = params
# The truth params = (;κ_soil = FT(1.5), ρc_soil = FT(2e6), f_bucket = FT(0.75), W_f = FT(0.2), p = FT(1), z_0m = FT(1e-2))
function bucket_turbulent_fluxes(params)
    diagnostics_outdir = joinpath(root_path, "global_diagnostics")
    outdir = ClimaUtilities.OutputPathGenerator.generate_output_path(
        diagnostics_outdir,
    )
    prob, cb = setup_prob(t0, tf, Δt, params, outdir; nelements)

    timestepper = CTS.RK4()
    ode_algo = CTS.ExplicitAlgorithm(timestepper)
    SciMLBase.solve(prob, ode_algo; dt = Δt, callback = cb, adaptive = false)

    simdir = ClimaAnalysis.SimDir(
        joinpath(root_path, "global_diagnostics", "output_active"),
    )
    lhf = get(simdir; short_name = "lhf")
    shf = get(simdir; short_name = "shf")

    # Initialize an empty list to store observations
    obs_list = []
    # Loop over each location
    for (lon, lat) in training_locations
        # Slice lhf and shf at the given longitude and latitude
        lhf_loc = ClimaAnalysis.slice(lhf, lon = lon, lat = lat)
        shf_loc = ClimaAnalysis.slice(shf, lon = lon, lat = lat)

        # Create Observation objects for lhf and shf
        lhf_obs = EKP.Observation(
            Dict(
                "samples" => lhf_loc.data,
                "covariances" => cov(lhf_loc.data) * EKP.I,
                "names" => "lhf_$(lon)_$(lat)",
            ),
        )
        shf_obs = EKP.Observation(
            Dict(
                "samples" => shf_loc.data,
                "covariances" => cov(shf_loc.data) * EKP.I,
                "names" => "shf_$(lon)_$(lat)",
            ),
        )

        # Add the observations to the list
        push!(obs_list, lhf_obs)
        push!(obs_list, shf_obs)
    end

    # Combine all observations into a single observation
    full_obs = EKP.combine_observations(obs_list)

    # Optional
    #=
    minibatcher = EKP.RandomFixedSizeMinibatcher(3) # batches the epoch of size 100, into batches of size 5
    observation_series = EKP.ObservationSeries(
    Dict(
    "observations" => full_obs,
    "minibatcher" => minibatcher,
    ),
    )
    =#

    obs = EKP.get_obs(full_obs)

    return full_obs, obs
end

# Read in the era5 datafile
era5_ds = Dataset(
    joinpath(
        ClimaLand.Artifacts.era5_surface_data2008_path(),
        "era5_monthly_surface_fluxes_200801-200812.nc",
    ),
)

# Make the ERA5 target
ERA5_target = []
close = (x, y) -> abs(x - y) < 5e-1
for (lon, lat) in training_locations
    # Fetch slices of lhf and shf era5 data from the era5 dataset
    lat_ind, lon_ind = findall((x) -> close(x, lat), era5_ds["latitude"][:])[1],
    findall((x) -> close(x, lon + 180), era5_ds["longitude"][:])[1]
    lhf_loc = vec(era5_ds["mslhf"][lon_ind, lat_ind, :][:, end, :])
    shf_loc = vec(era5_ds["msshf"][lon_ind, lat_ind, :][:, end, :])

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
