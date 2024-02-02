FROM python:3.10.12

WORKDIR /code

RUN apt-get update && apt-get install -y \
    git \
    bash-completion \
    htop \
    tmux \
    neovim

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# COPY .env_docker/sklearn_validation.py /usr/local/lib/python3.10/site-packages/sklearn/model_selection/_validation.py

CMD ["bash"]

