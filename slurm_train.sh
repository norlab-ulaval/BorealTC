#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00
#SBATCH --job-name=TerrainVulpi
#SBATCH --output=%x-%j.out

cd ~/Vulpi2021-terrain-deep-learning
docker build -t terrain-gpu -f DockerfileGPU .
container_id=$(
  docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
    --mount type=bind,source=.,target=/code/ \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    terrain-gpu python3 main.py
)

stop_container() {
  docker container stop $container_id
  docker logs $container_id
}

trap stop_container EXIT
echo "Container ID: $container_id"
docker wait $container_id
