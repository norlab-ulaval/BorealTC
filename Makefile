build:
	docker build -t terrain .

build-gpu:
	docker build -t terrain-gpu -f Dockerfile.gpu .

run: build
	docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
	  --mount type=bind,source="$(pwd)",target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  terrain python3 training.py
