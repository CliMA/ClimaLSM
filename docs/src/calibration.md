# Calibration framework of the CliMA land model

## Introduction

CliMA uses their own calibration pipeline for all their models ([ClimaLand](https://github.com/CliMA/ClimaLand.jl), [ClimaAtmos](https://github.com/CliMA/ClimaAtmos.jl), [ClimaOcean](https://github.com/CliMA/ClimaOcean.jl), [ClimaCoupler](https://github.com/CliMA/ClimaCoupler.jl)).
The core package of CliMA calibration is [EnsembleKalmanProcesses](https://github.com/CliMA/EnsembleKalmanProcesses.jl) (EKP),
which in a nutshell EnsembleKalmanProcesses (EKP) enables users to find an (locally-) optimal parameter set `u` for a computer code `G`
to fit some (noisy) observational data `y`.
It uses a suite of methods from the Ensemble Kalman filtering literature that have a long history of success in the weather forecasting community.
For more information, read [EKP documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/).

While it is possible to directly use EKP to calibrate our models, in practice we use [ClimaCalibrate](https://github.com/CliMA/ClimaCalibrate.jl),
which depends on EKP.jl, but further offer optimization for various HPC framework (e.g., central, clima, derecho), and an intuitive library and
output folder organization. For further information, read [ClimaCalibrate documentation](https://clima.github.io/ClimaCalibrate.jl/dev/).

## Calibrate a land model

The easiest, good first example to calibrate a land model with just EKP can be found in the
[tutorial to calibrate a single site latent heat flux](https://clima.github.io/ClimaLand.jl/stable/generated/calibration/minimal_working_example_obs/).

To calibrate a full land model globally with ClimaCalibrate, we will call a function that will look like this:
```julia
import ClimaCalibrate as CAL

CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    caldir,
)
```
where `CAL.WorkerBackend`, is one of the possible backend (others being, for example, `JuliaBackend`, `ClimaGPUBackend, or `DerechoBackend`),
see [this doc page](https://clima.github.io/ClimaCalibrate.jl/dev/backends/).
As explained, ClimaCalibrate can scale calibrations on different distributed computing environments, referred to as backends.
Each backend is optimized for specific use cases and computing resources. The backend system is implemented through Julia's multiple dispatch,
allowing seamless switching between different computing environments.

`prior` is the distribution of the parameters you want to calibrate. For example, if you want to calibrate two parameters called sc and pc,
you would define your priors like this, for example:
```julia
prior_sc = EKP.constrained_gaussian("sc", 5e-6, 5e-4, 0, Inf);
prior_pc = EKP.constrained_gaussian("pc", -2e6, 1e6, -Inf, Inf);
prior = EKP.combine_distributions([prior_sc, prior_pc]);
```
For more documentation about prior distribution, see [this EKP documentation page](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/).

`ensemble_size` is the number of parameter set drawn for your prior distribution tested at each iteration. For example, if you set ensemble_size to 5,
your model will be run 5 times (5 member) at each iteration, using 5 different parameter set drawn from your priors distribution. The distribution is
updated at each iteration by the EKP algorithm.

`n_iterations` is the number of times your priors distribution will be updated, at each iteration your model is run for the number of ensemble\_size.
So in total, your model will be run `ensemble_size` * `n_iterations`.

`observations` is the "truth" you want to calibrate your model on. It can take many forms.
For example, you may want to calibrate your land model latent heat flux (lhf), the
observations could be monthly global average of lhf, or monthly average at 100 random locations on land, or the annual amplitude and phase...
You will create `observations` from some data (for example ERA5), as a vector.

`noise` is the variance of each element of your `observations` vector. For example, if `observations` is monthly average of latent heat flux in a specific year,
your noise has to be the standard deviation of each month (over multiple years).

`caldir` is the path to your calibration output directory. For example `calibration_output`. Inside this folder, the parameter set of each iteration * member will
be stored, as well as the output of your model simulations. For example, if you ran a calibration with 1 iteration and 2 members, caldir would be structured
like this:

```
.
├── iteration_000
│   ├── eki_file.jld2
│   ├── G_ensemble.jld2
│   ├── member_001
│   │   ├── global_diagnostics
│   │   │   ├── output_0000
│   │   │   │   ├── lhf_1M_average.nc
│   │   │   │   ├── lwu_1M_average.nc
│   │   │   │   ├── shf_1M_average.nc
│   │   │   │   └── swu_1M_average.nc
│   │   │   └── output_active -> output_0000
│   │   └── parameters.toml
│   ├── member_002
│   │   ├── global_diagnostics
│   │   │   ├── output_0000
│   │   │   │   ├── lhf_1M_average.nc
│   │   │   │   ├── lwu_1M_average.nc
│   │   │   │   ├── shf_1M_average.nc
│   │   │   │   └── swu_1M_average.nc
│   │   │   └── output_active -> output_0000
│   │   └── parameters.toml
├── iteration_001
│   ├── eki_file.jld2
│   ├── G_ensemble.jld2
│   ├── member_001
│   │   ├── global_diagnostics
│   │   │   ├── output_0000
│   │   │   │   ├── lhf_1M_average.nc
│   │   │   │   ├── lwu_1M_average.nc
│   │   │   │   ├── shf_1M_average.nc
│   │   │   │   └── swu_1M_average.nc
│   │   │   └── output_active -> output_0000
│   │   └── parameters.toml
│   ├── member_002
│   │   ├── global_diagnostics
│   │   │   ├── output_0000
│   │   │   │   ├── lhf_1M_average.nc
│   │   │   │   ├── lwu_1M_average.nc
│   │   │   │   ├── shf_1M_average.nc
│   │   │   │   └── swu_1M_average.nc
│   │   │   └── output_active -> output_0000
│   │   └── parameters.toml
```
each iteration contains folders for each member, inside which you can find the parameters value inside parameters.toml,
and model outputs inside global_diagnostics.

Two additional functions need to be defined in order to run `CAL.calibrate`. `CAL.forward_model(iteration, member)` and `observation_map(iteration)`.
The `CAL.forward_model(iteration, member)` needs to generate your model output for a specific iteration and member. The `observation_map(iteration)`
needs to return your loss, a vector of the same format as `observations` but created with your model output (for example, monthly average of latent heat flux),
for all members. To make this easier, it can be useful to implement a `process_member_data(root_path)` function that generates one member output from your
model output path.

Once you have defined `CAL.forward_model`, `CAL.observation_map`, `caldir`, `noise`, `observations`, `n_iterations`, `ensemble_size`, and your backend, you can
call `CAL.calibrate`!

## job script

A calibration job will likely take hours to complete. You will probably have to submit a job from a HPC.
Below is an example job .pbs script, slurm would be slightly different.

```
#!/bin/bash
#PBS -N derecho_calibration
#PBS -o output.txt
#PBS -e error.txt
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=64:ngpus=4

## Account number for CliMA
#PBS -A UCIT0011
#PBS -q main

module load climacommon
# Run your command
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="SINGLETON"
julia --project=.buildkite -e 'using Pkg; Pkg.instantiate(;verbose=true)'

julia --project=.buildkite/ experiments/calibration/global_land/calibrate_land.jl
```

where `calibrate_land.jl` is a script that generates all the arguments needed and eventually calls `CAL.calibrate`.
You would start the job with a command such as `qsub name_of_job_script`, and a few hours later, you would get a calibrated parameter set.

