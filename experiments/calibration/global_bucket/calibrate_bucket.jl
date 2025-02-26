ENV["JULIA_WORKER_TIMEOUT"] = "300.0"

using ClimaLand
dir = pkgdir(ClimaLand)
import ClimaCalibrate: forward_model, parameter_path, path_to_ensemble_member
import ClimaCalibrate as CAL
using Distributed
addprocs(
    CAL.PBSManager(2),
    q = "main", # not prio
    A = "UCIT0011",
    l_select = "ngpus=4",
    l_walltime = "10:00:00",
)

@everywhere begin
    import ClimaCalibrate:
        forward_model, parameter_path, path_to_ensemble_member
    import ClimaCalibrate as CAL

    using ClimaLand
    caldir = "bucket_calibration_output"
    dir = pkgdir(ClimaLand)

    include(
        joinpath(dir, "experiments/calibration/global_bucket/forward_model.jl"),
    )
end

prior_κ_soil = EKP.constrained_gaussian("κ_soil", 2, 1, 0, Inf);
prior_ρc_soil = EKP.constrained_gaussian("ρc_soil", 4e6, 2e6, 0, Inf);
prior_f_bucket = EKP.constrained_gaussian("f_bucket", 0.6, 0.2, 0, 1);
prior_W_f = EKP.constrained_gaussian("W_f", 0.25, 0.2, 0, Inf);
prior_p = EKP.constrained_gaussian("p", 2, 1, 1, Inf);
prior_z_0m = EKP.constrained_gaussian("z_0m", 0.01, 0.04, 0, Inf);
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
noise = 1.0 * EKP.I # Should work, but this should be covariance of each month from observation (ERA5)

caldir = "bucket_calibration_output"
include(joinpath(dir, "experiments/calibration/shared/rand_locations.jl"))
include(
    joinpath(dir, "experiments/calibration/shared/make_training_locations.jl"),
)
include(
    joinpath(dir, "experiments/calibration/global_bucket/observation_map.jl"),
)
include(
    joinpath(
        dir,
        "experiments/calibration/shared/observation_data_locations.jl",
    ),
)

CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    caldir,
)
