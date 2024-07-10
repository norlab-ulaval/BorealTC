#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00
#SBATCH --job-name=TerrainMambaHusky
#SBATCH --output=%x-%j.out

cd ~/BorealTC
buildah build --layers -t borealtc .
container_id=$(
  podman run --device nvidia.com/gpu=all \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e DATASET='husky' --rm --ipc host \
    --mount type=bind,source=.,target=/code/ \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    -d borealtc python3 mamba_train.py
)

stop_container() {
  podman container stop $container_id
  podman logs $container_id
}

trap stop_container EXIT
echo "Container ID: $container_id"
podman wait $container_id
