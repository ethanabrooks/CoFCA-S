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
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


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
COPY . .
RUN pip install -e .
CMD nice exp \
  --cuda-deterministic \
  --log-dir=/home/ethanbro/tune_results
  --run-id=tune/maiden \
  --num-processes="300" \

  --tune \
  --redis-address 141.212.113.250:6379 \

  # env variables
  --subtask="AnswerDoor" \
  --subtask="AvoidDog" \
  --subtask="ComfortBaby" \
  --subtask="KillFlies" \
  --subtask="MakeFire" \
  --subtask="WatchBaby" \
  --test "WatchBaby"  "KillFlies" "MakeFire"\
  --test "AnswerDoor"  "KillFlies" "AvoidDog"\
  --test "AnswerDoor"  "MakeFire" "AvoidDog"\
  --n-active-subtasks="3" \
  --time-limit="30" \

  # intervals
  --eval-interval="100" \
  --log-interval="10" \
  --save-interval="300" \

  # dummies
  --env="" \
  --num-batch="-1" \
  --num-steps="-1" \
  --seed="-1" \
  --entropy-coef="-1" \
  --hidden-size="-1" \
  --num-layers="-1" \
  --learning-rate="-1" \
  --ppo-epoch="-1"
