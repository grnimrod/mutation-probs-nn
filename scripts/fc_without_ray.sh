#!/bin/bash
#SBATCH --job-name=fc_without_ray
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/%j_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/%j_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# Activate virtual environment
source /faststorage/project/MutationAnalysis/Nimrod/.venv/bin/activate

echo "which python: $(which python)"
echo "which python3: $(which python3)"

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/train_fc.py --data_version fA
