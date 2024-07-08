.PHONY: build podbuild run podrun podsh log watch-results jupyter

build:
	docker build -t borealtc .

podbuild:
	buildah build -t borealtc .

run: build
	docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
	  --mount type=bind,source=.,target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  borealtc python3 main.py

podrun: podbuild
	podman run --device nvidia.com/gpu=all --rm --ipc host \
	  --mount type=bind,source=.,target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  borealtc python3 main.py

podsh: podbuild
	podman run --device nvidia.com/gpu=all -it --rm --ipc host \
	  --mount type=bind,source=.,target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  borealtc

log:
	xdg-open http://localhost:6006 && tensorboard --logdir tb_logs

watch-results:
	watch "(ls results/husky/ && ls results/vulpi/) | wc"

jupyter: build
	@echo "Running jupyter-server"
	podman run -v .:/code -p 8887:8888 --rm borealtc \
	jupyter server --ip 0.0.0.0 --no-browser --NotebookApp.token='iros-2024' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --allow-root --ServerApp.notebook_dir=/code --ServerApp.root_dir=/code
