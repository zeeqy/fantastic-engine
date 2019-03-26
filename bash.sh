#!/bin/bash
#
#SBATCH --job-name=experiment
#SBATCH --output=res_output.txt  # output file
#SBATCH -e res_bug.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to 
#SBATCH --mem-per-cpu=4096   # Memory in MB per cpu allocated

echo $(date -u) "Main Engine Ignition!"

python3 mnist_baseline.py --seed=-1 --epochs=80 --burn_in=10 --reweight_interval=5 --num_cluster=6 --momentum=0.9 --valid_size=5000

echo $(date -u) "Mission Completed!"
