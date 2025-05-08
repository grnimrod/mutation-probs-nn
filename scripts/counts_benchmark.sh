#!/bin/bash
#SBATCH --job-name=counts_benchmark
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/counts_benchmark/%j_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/counts_benchmark/%j_error.log

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/train_counts_model.py --data_version 3fA
