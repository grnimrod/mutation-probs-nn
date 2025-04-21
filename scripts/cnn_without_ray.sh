#!/bin/bash
#SBATCH --job-name=cnn_without_ray
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/cnn/%j_cnn_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/cnn/%j_cnn_error.log

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/train_cnn.py --data_version fA
