#!/usr/bin/env zsh

pattern=${1:-'--active'}
tag="${2:-eval_rewards}"
correlate_cmd="runs correlate $pattern --value-path='.runs/logdir/<path>/10.$tag'"
echo $correlate_cmd
eval ${correlate_cmd}
