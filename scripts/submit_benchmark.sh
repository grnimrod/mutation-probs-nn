#!/bin/bash

versions=(3fA 3fC 5fA 5fC 7fA 7fC 9fA 9fC 11fA 11fC 13fA 13fC 15fA 15fC)

for version in "${versions[@]}"; do
    sbatch batch_benchmark.sh "$version"
done