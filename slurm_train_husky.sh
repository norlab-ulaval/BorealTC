#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00
#SBATCH --job-name=TerrainHusky
#SBATCH --output=%x-%j.out

cd ~/Vulpi2021-terrain-deep-learning
docker build -t terrain-gpu -f DockerfileGPU .
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -e DATASET='husky' --rm --ipc host \
  --mount type=bind,source=.,target=/code/ \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  terrain-gpu python3 main.py