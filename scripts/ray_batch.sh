#!/bin/bash
#SBATCH --job-name=ray_example
#SBATCH --nodes=2                # One node (modify for larger runs)
#SBATCH --ntasks=8               # One main task (Ray manages its own workers)
#SBATCH --cpus-per-task=1        # Number of available CPUs for Ray
#SBATCH --time=01:00:00          # Adjust as needed
#SBATCH --output=/faststorage/project/MutationAnalysis/Nimrod/results/logs/ray_output.log
#SBATCH --error=/faststorage/project/MutationAnalysis/Nimrod/results/logs/ray_error.log

# Activate virtual environment
source /faststorage/project/MutationAnalysis/Nimrod/.venv/bin/activate

# Get the node IP (needed for Ray)
HEAD_NODE=$(hostname -I | awk '{print $1}')

# Start Ray on the head node
ray start --head --node-ip-address="$HEAD_NODE" --port=6379 --num-cpus=$SLURM_CPUS_ON_NODE

# Wait a bit for Ray to stabilize
sleep 5

# Run the Python script
python /faststorage/project/MutationAnalysis/Nimrod/src/train_fc_ray.py
