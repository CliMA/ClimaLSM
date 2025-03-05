using ClimaAnalysis
import EnsembleKalmanProcesses as EKP

function process_member_data(simdir)

    lhf = get(simdir; short_name = "lhf")
    shf = get(simdir; short_name = "shf")
    lwu = get(simdir; short_name = "lwu")
    swu = get(simdir; short_name = "swu")

    vars_global_average = []
    for var in [lhf, shf, lwu, swu]
        var_global_average =
            ClimaAnalysis.average_lon(
                ClimaAnalysis.weighted_average_lat(
                    ClimaAnalysis.apply_oceanmask(var),
                ),
            ).data
        push!(vars_global_average, var_global_average)
    end

    return vcat(vars_global_average...)
end

function CAL.observation_map(iteration)
    single_member_dims = (4 * 12,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

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

    return G_ensemble
end
