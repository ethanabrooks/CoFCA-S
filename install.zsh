#!/usr/bin/env zsh

pip install --upgrade pip numpy cython ipython ipdb yapf isort pylint
repos=(mujoco rl-utils hsr-env gridworld-env lab-notebook)
for repo in $repos; do
  dir="../$repo"
  if [ ! -d $dir ]; then
    git clone "git@github.com:lobachevzky/$repo" "../$repo"
  fi 
  git --git-dir="$dir/.git" pull
  pip install -e "$dir"
done
pip install -e .
