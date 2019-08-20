FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7

# apt
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
         rsync \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN echo "conda init bash" > ~/.bashrc

# redis
RUN service redis-server start

# dependencies
RUN conda install -y -c pytorch pytorch 
RUN conda install -y pandas
COPY environment.yml .
RUN conda env create -f environment.yml
RUN git clone https://github.com/lobachevzky/rl-utils.git
RUN pip install -e rl-utils
RUN pip install \
      "ray[debug]==0.7.3" \ 
      "tensorboardX==1.8" \ 
      "tensorflow==1.14.0" \
      "opencv-python==4.1.0.25" \ 
      "psutil==5.6.3" \
      "ipython" \
      "ipdb"

RUN apt-get update && apt-get install -y --no-install-recommends net-tools vim iproute2
COPY . .
RUN pip install -e .
