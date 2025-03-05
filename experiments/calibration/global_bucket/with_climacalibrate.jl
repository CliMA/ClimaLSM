import ClimaCalibrate as CAL

# CAL.forward_model(iteration, member)
# --> just runs the model, doesn't return anything
# CAL.observation_map(iteration)
# --> returns the vector of obs

# @everywhere, see https://github.com/CliMA/ClimaAtmos.jl/blob/30ac26aafbb27c66bc65201dec5397116588a9af/calibration/test/e2e_test.jl

include("calibrate_bucket_function_climacalibrate.jl")

observations = full_obs_era5

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

ensemble_size = 10 # we could start with less
n_iterations = 5 # we could start with less
noise = EKP.I # not sure what to do here, I had noise embeded in my observation_map function

output_dir = "calibration_output" # Should I create this folder?

CAL.calibrate(
    CAL.WorkerBackend, # not sure what to do here
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    caldir, # to do: copy a tree of that folder structure
)
