#!/bin/bash
#SBATCH --job-name=fc_ray_trials
#SBATCH --account=MutationAnalysis
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/%j_ray_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/%j_ray_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

source /home/grnimrod/miniforge3/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mutation-probs-nn

# Get the node IP (needed for Ray)
HEAD_NODE=$(hostname -I | awk '{print $1}')

# Start Ray on the head node
ray start --head --node-ip-address="$HEAD_NODE" --port=6379 --num-cpus=$SLURM_CPUS_ON_NODE

# Wait a bit for Ray to stabilize
sleep 5

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/train_fc_ray.py --data_version fA
