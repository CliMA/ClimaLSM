#!/bin/bash
#SBATCH --job-name=calibrate_bucket
#SBATCH --output=output_%j.txt        # Output file (%j expands to job ID)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=5                    # Number of tasks (processes)
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --gpus-per-task=1

module load climacommon
# Run your command
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="SINGLETON"
julia --project=.buildkite -e 'using Pkg; Pkg.instantiate(;verbose=true)'

julia --project=.buildkite/ experiments/calibration/global_bucket/climacalibrate_bucket/calibrate.jl
