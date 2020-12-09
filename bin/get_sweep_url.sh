#! /usr/bin/env bash
wandb_output=$1
echo $wandb_output | tail -n2 | head -n1 | awk 'END {print $NF}'

