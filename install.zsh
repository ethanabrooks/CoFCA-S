#!/usr/bin/env zsh
set -e

repos=(rl-utils hsr-env gridworld-env lab-notebook ppo)
for repo in $repos; do
  dir="../$repo"
  if [ ! -d $dir ]; then
    git clone "git@github.com:lobachevzky/$repo" "../$repo"
  fi 
  cd $dir && git pull
  pip install -e .
done
