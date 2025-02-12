using ClimaLand
dir = pkgdir(ClimaLand)
import ClimaCalibrate: forward_model, parameter_path, path_to_ensemble_member
import ClimaCalibrate as CAL
using Distributed
addprocs(CAL.SlurmManager())

@everywhere begin
    import ClimaCalibrate: forward_model, parameter_path, path_to_ensemble_member
    import ClimaCalibrate as CAL

    using ClimaLand
    caldir = "calibration_output"
    dir = pkgdir(ClimaLand)

    include(joinpath(dir,"experiments/calibration/global_bucket/climacalibrate_bucket/bucket_target_script.jl"))
    include(joinpath(dir,"experiments/calibration/global_bucket/climacalibrate_bucket/forward_model.jl"))
end

include(joinpath(dir,"experiments/calibration/global_bucket/climacalibrate_bucket/observation_map.jl"))

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

ensemble_size = 10
n_iterations = 5
noise = 1.0*EKP.I # Should work, but this should be covariance of each month from observation (ERA5)

caldir = "calibration_output"

CAL.calibrate(
              CAL.WorkerBackend,
              ensemble_size,
              n_iterations,
              observations,
              noise,
              prior,
              caldir
             )
