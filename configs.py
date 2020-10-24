import copy
import json
from pathlib import Path

from hyperopt import hp

with Path(__file__).resolve().with_name("lower.json").open() as f:
    default_lower = json.load(f)

default = {
    "clip_param": 0.2,
    "cuda": True,
    "cuda_deterministic": False,
    "entropy_coef": 0.015,
    "eps": 1e-05,
    "eval_interval": None,
    "eval_steps": None,
    "gamma": 0.99,
    "hidden_size": 256,
    "learning_rate": 0.004,
    "load_path": None,
    "log_interval": 20000,
    "max_grad_norm": 0.5,
    "no_eval": False,
    "normalize": False,
    "num_batch": 2,
    "num_frames": 30000000,
    "num_layers": 1,
    "num_processes": 150,
    "ppo_epoch": 1,
    "recurrent": False,
    "render": False,
    "render_eval": False,
    "save_interval": 20000,
    "seed": 0,
    "synchronous": False,
    "tau": 0.95,
    "train_steps": 25,
    "use_gae": False,
    "value_loss_coef": 0.5,
}

search = copy.deepcopy(default)
search.update(
    entropy_coef=hp.choice("entropy_coef", [0.01, 0.015, 0.02]),
    hidden_size=hp.choice("hidden_size", [128, 256, 512]),
    learning_rate=hp.choice("learning_rate", [0.002, 0.003, 0.004]),
    num_batch=hp.choice("num_batch", [1, 2]),
    num_processes=hp.choice("num_processes", [50, 100, 150]),
    ppo_epoch=hp.choice("ppo_epoch", [1, 2, 3]),
    train_steps=hp.choice("train_steps", [20, 25, 30]),
    use_gae=hp.choice("use_gae", [True, False]),
)


default_upper = {
    "break_on_fail": False,
    "clip_param": 0.2,
    "conv_hidden_size": 64,
    "cuda": True,
    "cuda_deterministic": False,
    "debug": False,
    "debug_obs": False,
    "entropy_coef": 0.01,
    "env_id": "control-flow",
    "eps": 1e-5,
    "eval_interval": 1000000,
    "eval_steps": 500,
    "failure_buffer_size": 500,
    "fuzz": False,
    "gamma": 0.99,
    "gate_coef": 0.01,
    "hidden_size": 256,
    "inventory_hidden_size": 128,
    "kernel_size": 1,
    "learning_rate": 0.003,
    "load_path": None,
    "log_interval": 20000,
    "lower_embed_size": 64,
    "max_eval_lines": 50,
    "tgt_success_rate": 0.8,
    "max_grad_norm": 0.5,
    "max_lines": 10,
    "min_eval_lines": 1,
    "min_lines": 1,
    "no_eval": False,
    "no_pointer": False,
    "no_roll": False,
    "no_scan": False,
    "normalize": False,
    "num_batch": 1,
    "num_edges": 2,
    "num_frames": 200,
    "num_layers": 0,
    "num_processes": 150,
    "olsk": False,
    "ppo_epoch": 3,
    "recurrent": False,
    "render": False,
    "render_eval": False,
    "save_interval": 20000,
    "seed": 0,
    "sum_or_max": "sum",
    "stride": 1,
    "synchronous": False,
    "task_embed_size": 64,
    "tau": 0.95,
    "train_steps": 30,
    "transformer": False,
    "use_gae": False,
    "value_loss_coef": 0.5,
    "room_side": 4,
    "bridge_failure_prob": 0.1,
    "map_discovery_prob": 0.1,
    "bandit_prob": 0,
    "windfall_prob": 0,
    "debug_env": False,
    "exact_count": False,
    "hard_code_lower": False,
    # "bridge_failure_prob": 0.25,
    # "map_discovery_prob": 0.02,
    # "bandit_prob": 0.005,
    # "windfall_prob": 0.25,
}

search_upper = copy.deepcopy(default_upper)
search_upper.update(
    conv_hidden_size=hp.choice("conv_hidden_size", [32, 64]),
    entropy_coef=hp.choice("entropy_coef", [0.015, 0.02]),
    hidden_size=hp.choice("hidden_size", [512, 1024]),
    inventory_hidden_size=hp.choice("inventory_hidden_size", [256, 512]),
    kernel_size=hp.choice("kernel_size", [2]),
    learning_rate=hp.choice("learning_rate", [0.003, 0.004]),
    lower_embed_size=hp.choice("lower_embed_size", [32, 64, 128]),
    num_batch=hp.choice("num_batch", [1, 2]),
    num_edges=hp.choice("num_edges", [2, 4, 6]),
    ppo_epoch=hp.choice("ppo_epoch", [1, 2, 3]),
    tgt_success_rate=hp.choice("tgt_success_rate", [0.8, 0.9, 1]),
    task_embed_size=hp.choice("task_embed_size", [64, 128]),
    train_steps=hp.choice("train_steps", [25, 30, 35]),
    use_gae=hp.choice("use_gae", [True, False]),
)

search_lower = copy.deepcopy(default_lower)
search_lower.update(
    conv_hidden_size=hp.choice("conv_hidden_size", [32, 64]),
    entropy_coef=hp.choice("entropy_coef", [0.01, 0.015, 0.02]),
    hidden_size=hp.choice("hidden_size", [32, 64, 128, 256]),
    kernel_size=hp.choice("kernel_size", [2, 3, 4]),
    learning_rate=hp.choice("learning_rate", [0.002, 0.003, 0.004]),
    tgt_success_rate=hp.choice("tgt_success_rate", [0.5, 0.7, 0.8, 0.9, 1]),
    num_batch=hp.choice("num_batch", [1, 2]),
    num_conv_layers=1,
    num_layers=1,
    num_processes=150,
    ppo_epoch=hp.choice("ppo_epoch", [1, 2, 3]),
    stride=hp.choice("stride", [1, 2]),
    sum_or_max=hp.choice("sum_or_max", ["sum", "max"]),
    train_steps=hp.choice("train_steps", [20, 25, 30]),
    use_gae=hp.choice("use_gae", [True, False]),
)

search_debug = copy.deepcopy(search_upper)
search_debug.update(
    kernel_size=1,
    stride=1,
    world_size=1,
)
debug_default = copy.deepcopy(default_upper)
debug_default.update(
    kernel_size=1,
    stride=1,
    world_size=1,
)
configs = dict(
    search_lower=search_lower,
    search_upper=search_upper,
    search_debug=search_debug,
    default_upper=default_upper,
    debug_default=debug_default,
    search=search,
)
