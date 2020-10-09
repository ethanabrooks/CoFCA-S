FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN apt-get update && apt-get install -y rsync  && rm -rf /var/lib/apt/lists/*

COPY ./environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean --all -y \
    && echo "source activate teacher" >> /root/.bashrc
SHELL ["conda", "run", "-n", "teacher", "/bin/bash", "-c"]
RUN pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp38-cp38-manylinux1_x86_64.whl
RUN pip install ipdb

WORKDIR "/root"
COPY . .
