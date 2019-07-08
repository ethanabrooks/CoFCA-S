#!/usr/bin/env zsh
sed 's/batch-size="\{0,1\}[^ "]*"\{0,1\}/batch-size=1/' |\
sed 's/num-processes="\{0,1\}[^ "]*"\{0,1\}/num-processes=1/'
