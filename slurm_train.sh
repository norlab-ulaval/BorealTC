#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=4
#SBATCH --time=4-00:00
#SBATCH --job-name=TerrainCNN
#SBATCH --output=%x-%j.out

cd ~/Vulpi2021-terrain-deep-learning
docker build -t terrain -f DockefileGPU .
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
  --mount type=bind,source="$(pwd)",target=/code/ \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  terrain python3 training.py
