#!/bin/bash
#SBATCH --job-name=train_without_ray
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/output_%j.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=2 # Modify for larger runs
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# Activate virtual environment
source /faststorage/project/MutationAnalysis/Nimrod/.venv/bin/activate

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/train_fc.py
