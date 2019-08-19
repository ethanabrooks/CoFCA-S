FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libglib2.0-0 \
         libsm6 \
         libxext6 \
         libxrender-dev \
         redis-server \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN service redis-server start
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN echo "conda init bash" > ~/.bashrc
WORKDIR /ppo
RUN git clone https://github.com/lobachevzky/rl-utils.git
RUN pip install -e rl-utils
RUN conda install -y pandas
RUN conda install -y -c pytorch pytorch 
COPY environment.yml .
RUN conda env create -f environment.yml
COPY setup.py .
RUN pip install \
      "ray[debug]==0.7.3" \ 
      "tensorboardX==1.8" \ 
      "opencv-python==4.1.0.25" \ 
      "psutil==5.6.3"
COPY . .
RUN pip install -e .
