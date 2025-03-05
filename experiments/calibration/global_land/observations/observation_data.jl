using NCDatasets
import EnsembleKalmanProcesses as EKP
using ClimaUtilities.ClimaArtifacts
import ClimaLand
using StaticArrays

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

# Define the locations longitudes and latitudes
# Needs to be defined once, both for g(θ) and ERA5 target
t0 = 0.0
tf = 60 * 60.0 * 24 * 366
Δt = 900.0
nelements = (50, 10)
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
    lwu_loc = vec(era5_ds["msuwlwrf"][lon_ind, lat_ind, :][:, end, :])
    swu_loc = vec(era5_ds["msuwswrf"][lon_ind, lat_ind, :][:, end, :])

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

    lwu_ERA5 = EKP.Observation(
        Dict(
            "samples" => lwu_loc,
            "covariances" => cov(lwu_loc) * EKP.I,
            "names" => "lwu_$(lon)_$(lat)",
        ),
    )
    swu_ERA5 = EKP.Observation(
        Dict(
            "samples" => swu_loc,
            "covariances" => cov(swu_loc) * EKP.I,
            "names" => "swu_$(lon)_$(lat)",
        ),
    )

    # Add the observations to the target
    push!(ERA5_target, lhf_ERA5)
    push!(ERA5_target, shf_ERA5)
    push!(ERA5_target, lwu_ERA5)
    push!(ERA5_target, swu_ERA5)
end

observations = EKP.get_obs(EKP.combine_observations(ERA5_target))
