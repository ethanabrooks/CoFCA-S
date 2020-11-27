FROM nvcr.io/nvidia/pytorch:19.10-py3

#RUN apt-get update && apt-get install -y rsync  && rm -rf /var/lib/apt/lists/*

COPY ./environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean --all -y \
    && echo "source activate ppo" >> /root/.bashrc
SHELL ["conda", "run", "-n", "ppo", "/bin/bash", "-c"]

VOLUME ["/ppo"]
WORKDIR "/ppo"
#COPY entrypoint.sh /ppo/
#ENTRYPOINT ["/ppo/entrypoint.sh"]
