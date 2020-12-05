#! /usr/bin/env bash

runs_per_gpu=4
ngpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nruns=$(($ngpu * $runs_per_gpu))
session='session'


while getopts n:s:i: flag
do
    case "${flag}" in
      n) nruns=${OPTARG};;
      s) session=${OPTARG};;
      i) id=${OPTARG};;
      *) echo "args are -n and -i" && exit;;
    esac
done

echo "Creating $nruns sessions..."

for i in $(seq 1 $nruns); do
  gpu=$(($i % $ngpu));
  echo "tmux at -t $session$i"
  tmux new-session -d -s "$session$i" "CUDA_VISIBLE_DEVICES=$gpu wandb agent $id"
  #echo docker run \
    #--rm \
    #--detach \
    #--gpus $gpu \
    #--volume $(pwd):/ppo \
    #--env WANDB_API_KEY=$key \
    #ethanabrooks/ppo $id
done
