#!/bin/bash
#SBATCH --job-name=calibrate_bucket
#SBATCH --output=output_%j.txt        # Output file (%j expands to job ID)
#SBATCH --error=error_%j.txt          # Error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --time=05:00:00               # Time limit hrs:min:sec
#SBATCH --mem=8G                      # Memory per node
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Run your command
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="SINGLETON"

julia --project=.buildkite/ experiments/calibration/global_bucket/calibrate_bucket.jl

