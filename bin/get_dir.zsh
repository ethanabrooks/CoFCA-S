#!/usr/bin/env zsh

pattern=${1-'--active'}
sessions=(${(f)"$(runs ls $pattern)"})
echo "${sessions[1]%/[0-9]*}"
