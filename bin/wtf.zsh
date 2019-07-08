#!/usr/bin/env zsh
if [[ "$1" =~ '[0-9]+' && "$1" = $MATCH ]]; then
  echo "wtf ppo/main ${@:(($1 + 2))}"
else
  echo "$@"
fi
