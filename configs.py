import copy
from pathlib import Path

from hyperopt import hp

from lines import Subtask, If, Else, While

default = dict(
    control_flow_types=[Subtask, If, While, Else],
    conv_hidden_size=64,
    entropy_coef=0.015,
    eval_interval=100,
    eval_steps=500,
    failure_buffer_size=500,
    gate_coef=0.01,
    hidden_size=256,
    inventory_hidden_size=128,
    kernel_size=2,
    learning_rate=0.003,
    lower_embed_size=64,
    lower_level_config=Path("checkpoint/lower.json"),
    lower_level_load_path=Path("checkpoint/lower.pt"),
    max_eval_lines=50,
    max_failure_sample_prob=0.3,
    max_lines=10,
    max_loops=3,
    max_nesting_depth=1,
    max_while_loops=10,
    max_world_resamples=50,
    min_eval_lines=1,
    min_lines=1,
    no_op_coef=0,
    no_op_limit=30,
    num_batch=1,
    num_edges=2,
    num_layers=0,
    num_processes=150,
    ppo_epoch=2,
    reject_while_prob=0.6,
    stride=1,
    task_embed_size=64,
    term_on=["mine", "sell"],
    time_to_waste=0,
    train_steps=25,
    world_size=6,
)

search = copy.deepcopy(default)
search.update(
    conv_hidden_size=hp.choice("conv_hidden_size", [32, 64, 128]),
    entropy_coef=hp.choice("entropy_coef", [0.01, 0.015, 0.02]),
    gate_coef=hp.choice("gate_coef", [0, 0.01, 0.05]),
    hidden_size=hp.choice("hidden_size", [128, 256, 512]),
    inventory_hidden_size=hp.choice("inventory_hidden_size", [64, 128, 256]),
    kernel_size=hp.choice("kernel_size", [1, 2, 3]),
    learning_rate=hp.choice("learning_rate", [0.002, 0.003, 0.004]),
    lower_embed_size=hp.choice("lower_embed_size", [32, 64, 128]),
    max_failure_sample_prob=hp.choice("max_failure_sample_prob", [0.2, 0.3, 0.4]),
    max_while_loops=hp.choice("max_while_loops", [5, 10, 15]),
    no_op_limit=hp.choice("no_op_limit", [20, 30, 40]),
    num_batch=hp.choice("num_batch", [1, 2]),
    num_edges=hp.choice("num_edges", [2, 4, 6]),
    num_processes=hp.choice("num_processes", [50, 100, 150]),
    ppo_epoch=hp.choice("ppo_epoch", [1, 2, 3]),
    reject_while_prob=hp.choice("reject_while_prob", [0.5, 0.6]),
    stride=hp.choice("stride", [1, 2, 3]),
    task_embed_size=hp.choice("task_embed_size", [32, 64, 128]),
    train_steps=hp.choice("train_steps", [20, 25, 30]),
)

debug_search = copy.deepcopy(search)
debug_search.update(
    kernel_size=1, stride=1, world_size=1,
)
del debug_search["lower_level_config"]
del debug_search["lower_level_load_path"]
configs = dict(search=search, debug_search=debug_search, default=default)
