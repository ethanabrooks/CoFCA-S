#!/usr/bin/env zsh
sed 's/log-dir=\("\{0,1\}\)\([^ "]*\)\("\{0,1\}\)/load-path=\1\2\/checkpoint.pt\3/'
