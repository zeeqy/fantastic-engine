#!/bin/bash
#
#SBATCH --job-name=experiment
#SBATCH --output=res_output.txt  # output file
#SBATCH -e res_bug.err        # File to which STDERR will be written
#SBATCH --partition=1080ti-short # Partition to submit to 
#SBATCH --mem-per-cpu=4000   # Memory in MB per cpu allocated

echo $(date -u) "Main Engine Ignition!"

python3 MNIST_Baseline.py --seed=-1 --epochs=5 --burn_in=3

echo $(date -u) "Mission Completed!"
