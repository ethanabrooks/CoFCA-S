#!/usr/bin/env zsh

session="${${(f)"$(tmux ls)"}[(r)$1]}"
echo "${session%/[0-9]*}"
echo $dir
