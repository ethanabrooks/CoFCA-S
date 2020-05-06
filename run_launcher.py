#! /usr/bin/env python
import json
import random
import argparse


p = argparse.ArgumentParser()
p.add_argument("num", type=int)
p.add_argument("name")
args = p.parse_args()

config_map = {
    f"{args.name}{i}": f"""\
--gridworld \
--control-flow-types Subtask \
--conv-hidden-size={random.choice([128, 256])} \
--entropy-coef=0.015 \
--eval-interval=100 \
--eval-lines=1 \
--eval-steps=500 \
--flip-prob=0.5 \
--gate-coef=0.01 \
--gate-conv-kernel-size=3 \
--gate-hidden-size=32 \
--gate-pool-kernel-size=3 \
--gate-pool-stride=2 \
--gru-gate-coef=0 \
--gru-hidden-size=64 \
--hidden-size=64 \
--kernel-size=2 \
--learning-rate=0.0035 \
--log-dir=/log \
--lower-level-config=checkpoint/lower.json \
--lower-level=train-with-upper \
--max-lines=1 \
--max-loops=3 \
--max-nesting-depth=1 \
--max-while-loops=7 \
--max-world-resamples=5 \
--min-lines=1 \
--no-op-coef=0 \
--no-op-limit=30 \
--num-batch=1 \
--num-conv-layers=1 \
--num-edges=3 \
--num-encoding-layers=0 \
--num-layers=0 \
--num-processes=150 \
--num-steps=25 \
--ppo-epoch=2 \
--run-id=control-flow/train-with-upper/subtask/train1/eval1/10 \
--seed=1 \
--stride=2 \
--term-on=mine \
--time-to-waste=0 \
--world-size=6 \
"""
    for i in range(args.num)
}

with open("/tmp/config_map.json", "w") as f:
    json.dump(config_map, f)
