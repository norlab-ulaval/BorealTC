build:
	docker build -t terrain .

build-gpu:
	docker build -t terrain-gpu -f Dockerfile.gpu .

run: build
	docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
	  --mount type=bind,source="$(pwd)",target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  terrain python3 training.py

log:
	xdg-open http://localhost:6006 && tensorboard --logdir .

jupyter: build
	@echo "Running jupyter-server"
	docker run -v .:/code -p 8887:8888 --rm terrain \
	jupyter server --ip 0.0.0.0 --no-browser --NotebookApp.token='iros-2024' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --allow-root --ServerApp.notebook_dir=/code --ServerApp.root_dir=/code
