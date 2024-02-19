build:
	docker build -t terrain .

run: build
	docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
	  --mount type=bind,source=.,target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  terrain python3 main.py

build-gpu:
	docker build -t terrain-gpu -f DockerfileGPU .

run-gpu: build-gpu
	docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
	  --mount type=bind,source=.,target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  terrain-gpu python3 main.py

log:
	xdg-open http://localhost:6006 && tensorboard --logdir .

watch-results:
	watch "(ls results/husky/ && ls results/vulpi/) | wc"

jupyter: build
	@echo "Running jupyter-server"
	docker run -v .:/code -p 8887:8888 --rm terrain \
	jupyter server --ip 0.0.0.0 --no-browser --NotebookApp.token='iros-2024' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --allow-root --ServerApp.notebook_dir=/code --ServerApp.root_dir=/code
