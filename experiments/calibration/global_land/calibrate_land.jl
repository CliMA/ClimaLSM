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
    caldir = "calibration_output"
    dir = pkgdir(ClimaLand)

    include(
        joinpath(
            dir,
            "experiments/calibration/global_land/forward_model.jl",
        ),
    )
end

import EnsembleKalmanProcesses as EKP

## Priors
prior_pc = EKP.constrained_gaussian("pc", -2e6, 1e6, -Inf, 0);
prior_sc = EKP.constrained_gaussian("sc", 5e-6, 5e-4, 0, Inf);
prior = EKP.combine_distributions([prior_pc, prior_sc]);

ensemble_size = 6
n_iterations = 2
noise = 1.0 * EKP.I # Should work, but this should be covariance of each month from observation (ERA5)

caldir = "calibration_output"

include(
    joinpath(
        dir,
        "experiments/calibration/global_land/observation_map.jl",
    ),
)

include(joinpath(dir, "experiments/calibration/shared/observation_data_global.jl"))

CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    caldir,
)
