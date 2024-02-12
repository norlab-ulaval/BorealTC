#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=4
#SBATCH --time=4-00:00
#SBATCH --job-name=TerrainVulpi
#SBATCH --output=%x-%j.out

cd ~/Vulpi2021-terrain-deep-learning
make build-gpu
make run-gpu
