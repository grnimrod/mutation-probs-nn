#!/bin/bash
#SBATCH --job-name=combined
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/combined/%j_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/combined/%j_error.log

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

# Run the Python script
# python /faststorage/project/MutationAnalysis/Nimrod/src/train_combined_avg_mut.py --data_version 3fA

SCRIPT_PATH=$1
DATA_VERSION=$2

# Run training
python "$SCRIPT_PATH" --data_version "$DATA_VERSION"