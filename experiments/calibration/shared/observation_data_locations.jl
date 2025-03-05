using NCDatasets

era5_ds = Dataset(
    joinpath(
        ClimaLand.Artifacts.era5_surface_data2008_path(),
        "era5_monthly_surface_fluxes_200801-200812.nc",
    ),
)

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


# Make the ERA5 target
ERA5_target = []
ERA5_std = []
close_location = (x, y) -> abs(x - y) < 5e-1
for (lon, lat) in training_locations # locations
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
    push!(ERA5_std, lhf_ERA5_std)
    push!(ERA5_std, shf_ERA5_std)
    return nothing
end

full_obs_era5 = EKP.combine_observations(ERA5_target)
observations = EKP.get_obs(full_obs_era5)
noise_era5 = ERA5_std
# to do: convert ERA5_std to a 24-element Vector
# possibly need to change sign of lhf and shf *-1

# to do, use climaanalysis instead of netcdf,
# make sure same grid is used
# and timestamps align
#=

filepath =joinpath(
           ClimaLand.Artifacts.era5_surface_data2008_path(),
           "era5_monthly_surface_fluxes_200801-200812.nc",
       )
data = OutputVar(filepath)
# lat and lon needs to be defined the same for model and era5
# this can be done with the function below (but it needs to be fixed)
# center_longitude!(data, 180)
slice(data, latitude = 0, longitude = 45)


# to ensure observation (era5) and model align in time, use the same approach as
# https://github.com/CliMA/ClimaLand.jl/blob/main/experiments/long_runs/leaderboard/data_sources.jl#L95

=#

