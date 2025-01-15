# Use the .buildkite environment
# bucket_turbulent_fluxes(params) returns lhf and shf as a function of 6 parameters
include("calibrate_bucket_function.jl")

using Random

#= Target (perfect model)
params = (;
    κ_soil = FT(1.5),
    ρc_soil = FT(2e6),
    f_bucket = FT(0.75),
    W_f = FT(0.2),
    p = FT(1),
    z_0m = FT(1e-2),
)
target = bucket_turbulent_fluxes(params)[1]
=#

target = full_obs_era5

# Parameters prior
prior_κ_soil = EKP.constrained_gaussian("κ_soil", 2, 1, 0, Inf);
prior_ρc_soil = EKP.constrained_gaussian("ρc_soil", 4e6, 2e6, 0, Inf);
prior_f_bucket = EKP.constrained_gaussian("f_bucket", 0.5, 0.3, 0, 1);
prior_W_f = EKP.constrained_gaussian("W_f", 0.4, 0.4, 0, Inf);
prior_p = EKP.constrained_gaussian("p", 2, 1, 1, Inf);
prior_z_0m = EKP.constrained_gaussian("z_0m", 0.01, 0.1, 0, Inf);
prior = EKP.combine_distributions([
    prior_κ_soil,
    prior_ρc_soil,
    prior_f_bucket,
    prior_W_f,
    prior_p,
    prior_z_0m,
]);

rng_seed = 2
rng = Random.MersenneTwister(rng_seed)

N_ensemble = 10
N_iterations = 5

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble);
ensemble_kalman_process = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    target,
    EKP.Inversion();
    rng = rng,
);

for i in 1:N_iterations
    params_i = EKP.get_ϕ_final(prior, ensemble_kalman_process)

    G_ens = hcat(
        [
            bucket_turbulent_fluxes((;
                κ_soil = params_i[1, j],
                ρc_soil = params_i[2, j],
                f_bucket = params_i[3, j],
                W_f = params_i[4, j],
                p = params_i[5, j],
                z_0m = params_i[6, j],
            ))[2] for j in 1:N_ensemble
        ]...,
    )

    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

final_ensemble = EKP.get_ϕ_final(prior, ensemble_kalman_process)
println(final_ensemble)
