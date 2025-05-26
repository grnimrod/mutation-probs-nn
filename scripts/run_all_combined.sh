#!/bin/bash

scripts=(
    "/faststorage/project/MutationAnalysis/Nimrod/src/train_combined_avg_mut.py"
    "/faststorage/project/MutationAnalysis/Nimrod/src/train_combined_bin_id.py"
    "/faststorage/project/MutationAnalysis/Nimrod/src/train_combined_bin_norm.py"
    "/faststorage/project/MutationAnalysis/Nimrod/src/train_combined_avgmut_binidnorm.py"
    "/faststorage/project/MutationAnalysis/Nimrod/src/train_combined_allthree.py"
)

versions=(3fA 3fC 5fA 5fC 7fA 7fC 9fA 9fC 11fA 11fC 13fA 13fC 15fA 15fC)

for script in "${scripts[@]}"; do
  for version in "${versions[@]}"; do
    sbatch combined.sh "$script" "$version"
  done
done