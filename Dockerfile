FROM docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set the current timezone
ENV TZ=America/Toronto \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    # CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    # CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE \
    # CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python and dependencies
RUN apt-get update && apt-get -y install software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y git ninja-build nvidia-cuda-toolkit libjpeg-dev zlib1g-dev libopenblas-dev
RUN apt-get install -y python3 python3-dev python3-venv python3-pip

RUN apt-get update && apt-get install -y \
    git \
    bash-completion \
    htop \
    tmux \
    neovim
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

WORKDIR /code

# Mamba compilation
RUN pip3 install packaging torch torchvision wheel

COPY requirements-lock.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install mamba-ssm==1.2.0
ENV PYTHONPATH=/code:$PYTHONPATH

CMD ["bash"]
