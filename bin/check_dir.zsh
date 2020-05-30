#!/usr/bin/env zsh

prompt="Is this the correct directory path?:
$1
"
vared -p $prompt -c answer
if [[ $answer = n* ]]; then
  exit 1
fi 
