#!/bin/bash
#SBATCH --job-name=cnn_without_ray
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/output_%j.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

echo "which python: $(which python)"

# Run the Python script
# python /faststorage/project/MutationAnalysis/Nimrod/src/train_cnn.py --data_version sA
