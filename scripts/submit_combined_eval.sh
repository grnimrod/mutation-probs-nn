#!/bin/bash
#SBATCH --job-name=evaluate_combined
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/evaluate/%j_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/evaluate/%j_error.log
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/evaluate_models.py