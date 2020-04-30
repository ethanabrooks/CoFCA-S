#! /usr/bin/env bash
pip install -e .
python ppo/control_flow/main.py $@
