using ClimaAnalysis
import EnsembleKalmanProcesses as EKP
#nonan_training_locations = SVector(deleteat!(collect(training_locations), rows_to_remove)...)
#rows_to_remove = [41, 59, 70] # iteration 0, most in member 5. member 1 has no nan.
#rows_to_remove = [66] # iteration 1, member 5

# explain things here - tutorial style
function process_member_data(simdir)

    lhf = get(simdir; short_name = "lhf")
    shf = get(simdir; short_name = "shf")
    lwu = get(simdir; short_name = "lwu")
    swu = get(simdir; short_name = "swu")

#    rows_to_remove=[]
    # Initialize an empty list to store observations
    obs_list = []
#    i=1
    # Loop over each location
    for (lon, lat) in nonan_training_locations # training_locations
        # Slice lhf and shf at the given longitude and latitude
        lhf_loc = ClimaAnalysis.slice(lhf, lon = lon, lat = lat)
        shf_loc = ClimaAnalysis.slice(shf, lon = lon, lat = lat)
        lwu_loc = ClimaAnalysis.slice(lwu, lon = lon, lat = lat)
        swu_loc = ClimaAnalysis.slice(swu, lon = lon, lat = lat)

        # Create Observation objects for lhf and shf
        lhf_obs = EKP.Observation(
            Dict(
                 "samples" => lhf_loc.data[3:12],
                "covariances" => cov(lhf_loc.data[3:12]) * EKP.I,
                "names" => "lhf_$(lon)_$(lat)",
            ),
        )
        shf_obs = EKP.Observation(
            Dict(
                "samples" => shf_loc.data[3:12],
                "covariances" => cov(shf_loc.data[3:12]) * EKP.I,
                "names" => "shf_$(lon)_$(lat)",
            ),
        )

        lwu_obs = EKP.Observation(
            Dict(
                "samples" => lwu_loc.data[3:12],
                "covariances" => cov(lwu_loc.data[3:12]) * EKP.I,
                "names" => "lwu_$(lon)_$(lat)",
            ),
        )

        swu_obs = EKP.Observation(
            Dict(
                "samples" => swu_loc.data[3:12],
                "covariances" => cov(swu_loc.data[3:12]) * EKP.I,
                "names" => "swu_$(lon)_$(lat)",
            ),
        )
        # Add the observations to the list
        #if all(x -> all(!isnan, x), [lhf_loc.data, shf_loc.data, lwu_loc.data, swu_loc.data])
            push!(obs_list, lhf_obs)
            push!(obs_list, shf_obs)
            push!(obs_list, lwu_obs)
            push!(obs_list, swu_obs)
        #end

#        if all(x -> all(!isnan, x), [lhf_loc.data, shf_loc.data, lwu_loc.data, swu_loc.data])
#            println("no nan", ", m = ", m, ", i = ", i)
#        else
#            println("some nans", ", m = ", m, ", i = ", i)
#            push!(rows_to_remove, i)
#        end
#        i+=1
    end

    # Combine all observations into a single observation
    full_obs = EKP.combine_observations(obs_list)
    obs = EKP.get_obs(full_obs)
    return obs
end

function CAL.observation_map(iteration)
    single_member_dims = 4*10*83
    #single_member_dims = (912,) # defined in calibrate_land.jl
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

#    rows_to_remove = []
    for m in 1:ensemble_size
        member_path = path_to_ensemble_member(caldir, iteration, m)
        simdir_path =
            joinpath(member_path, "global_diagnostics", "output_active")
        if isdir(simdir_path)
            simdir = SimDir(simdir_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        else
            G_ensemble[:, m] .= NaN
        end
    end

#    rows_to_remove = unique(rows_to_remove)
    # now in G_ensemble, remove the locations (12 rows for each index of rows_to_remove)
    # then remove those for observation_data too
    return G_ensemble
end
