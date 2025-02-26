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
            "samples" => -lhf_loc,
            "covariances" => cov(lhf_loc) * EKP.I,
            "names" => "lhf_$(lon)_$(lat)",
        ),
    )

    shf_ERA5 = EKP.Observation(
        Dict(
            "samples" => -shf_loc,
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
