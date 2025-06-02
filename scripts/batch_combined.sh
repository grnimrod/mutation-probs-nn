#!/bin/bash
#SBATCH --job-name=combined
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/combined/%j_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/combined/%j_error.log
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

DATA_VERSION=$1

# Run training
python /faststorage/project/MutationAnalysis/Nimrod/src/train_combined.py --data_version "$DATA_VERSION"