#! /usr/bin/env bash

conda activate ppo
export CONTAINER_VOLUME="/root/ray_results"
export RUN_KILL_LABEL="RUN"
export RUN_DB_PATH="$HOME/runs/ppo"
export RUN_IMAGE_BUILD_PATH="$(realpath .)"
export RUN_DOCKERFILE_PATH="$RUN_IMAGE_BUILD_PATH/Dockerfile"
export RUN_IMAGE='ppo'
export RUN_CONFIG_SCRIPT='config_script.py'
export RUN_CONFIG_SCRIPT_INTERPRETER='python3'
export RUN_CONFIG_SCRIPT_INTERPRETER_ARGS='-c'
export DOCKER_RUN_COMMAND='docker run --rm --gpus all -it --detach --label RUN'
