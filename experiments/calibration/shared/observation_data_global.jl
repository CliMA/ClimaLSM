import ClimaAnalysis
using ClimaLand
using ClimaLand.ClimaArtifacts

filename = joinpath(
    ClimaLand.Artifacts.era5_surface_data2008_path(),
    "era5_monthly_surface_fluxes_200801-200812.nc",
)

era5_lhf = ClimaAnalysis.OutputVar(filename, "mslhf")
era5_shf = ClimaAnalysis.OutputVar(filename, "msshf")
era5_lwu = ClimaAnalysis.OutputVar(filename, "msuwlwrf")
era5_swu = ClimaAnalysis.OutputVar(filename, "msuwswrf")

observations_vecs = []
for var in [era5_lhf, era5_shf, era5_lwu, era5_swu]
    var_global_average =
        ClimaAnalysis.average_lon(
            ClimaAnalysis.weighted_average_lat(
                ClimaAnalysis.apply_oceanmask(var),
            ),
        ).data
    push!(observations_vecs, var_global_average)
end
observations = vcat(observations_vecs...)
observations[1:24] .*= -1 # different sign convention for lhf and shf
