#!/usr/bin/env zsh

pip install --upgrade pip numpy cython ipython ipdb yapf isort pylint
repos=(mujoco rl-utils hsr-env gridworld-env lab-notebook)
here=$(pwd)
for repo in $repos; do
  dir="../$repo"
  if [ ! -d $dir ]; then
    git clone "git@github.com:lobachevzky/$repo" "../$repo"
  fi 
  cd $dir && git pull
  pip install -e .
done
pip install -e $here
