#! /usr/bin/env bash

runs_per_gpu=4
ngpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nruns=$(($ngpu * $runs_per_gpu))
session='session'

while getopts c:n:r:s: flag
do
    case "${flag}" in
      c) config=${OPTARG};;
      n) name=${OPTARG};;
      r) nruns=${OPTARG};;
      s) session=${OPTARG};;
      *) echo "usage: run.sh -c <config> -n <name> -r <nruns> -s <session>" && exit;;
    esac
done

wandb_output=$(wandb sweep --name "$n" "$config")
url=$(${0:a:h}/bin/get_sweep_url.sh $wandb_output)
id=$(${0:a:h}/bin/get_agent_id.sh $wandb_output)
echo "wandb: View sweep at: $url"

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
