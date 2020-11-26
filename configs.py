import copy

from wandb.sweeps.config.hyperopt import hp

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
    "log_interval": 37500,
    "max_grad_norm": 0.5,
    "no_eval": True,
    "normalize": False,
    "num_batch": 2,
    "num_frames": 30000000,
    "num_layers": 1,
    "num_processes": 150,
    "ppo_epoch": 1,
    "recurrent": False,
    "render": False,
    "render_eval": False,
    "save_interval": 37500,
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


starcraft_default = {
    "break_on_fail": False,
    "clip_param": 0.2,
    "conv_hidden_size": 64,
    "cuda": True,
    "cuda_deterministic": False,
    "debug": False,
    "debug_env": False,
    "destroy_building_prob": 0.01,
    "entropy_coef": 0.01,
    "env_id": "control-flow",
    "eps": 1e-5,
    "eval_interval": 1000000,
    "eval_steps": 500,
    "failure_buffer_size": 500,
    "gamma": 0.99,
    "gate_coef": 0.01,
    "hidden_size": 256,
    "resources_hidden_size": 128,
    "kernel_size": 2,
    "learning_rate": 0.003,
    "load_path": None,
    "log_interval": 37500,
    "lower_embed_size": 64,
    "tgt_success_rate": 0.8,
    "max_grad_norm": 0.5,
    "max_eval_lines": 50,
    "max_lines": 10,
    "min_eval_lines": 1,
    "min_lines": 1,
    "next_actions_embed_size": 64,
    "no_eval": True,
    "no_pointer": False,
    "no_roll": False,
    "no_scan": False,
    "normalize": False,
    "num_batch": 1,
    "num_edges": 2,
    "num_frames": 5000000,
    "num_initial_buildings": 2,
    "num_layers": 0,
    "num_processes": 150,
    "olsk": False,
    "ppo_epoch": 3,
    "recurrent": False,
    "render": False,
    "render_eval": False,
    "save_interval": 37500,
    "seed": 0,
    "stride": 1,
    "synchronous": False,
    "task_embed_size": 128,
    "tau": 0.95,
    "train_steps": 30,
    "transformer": False,
    "use_gae": False,
    "value_loss_coef": 0.5,
}

search_starcraft = copy.deepcopy(starcraft_default)
search_starcraft.update(
    conv_hidden_size=hp.choice("conv_hidden_size", [32, 64]),
    entropy_coef=hp.choice("entropy_coef", [0.02, 0.025]),
    hidden_size=hp.choice("hidden_size", [64, 128, 256]),
    learning_rate=hp.choice("learning_rate", [0.001, 0.0015, 0.002, 0.0025]),
    lower_embed_size=hp.choice("lower_embed_size", [32, 64]),
    next_actions_embed_size=hp.choice("next_actions_embed_size", [32, 64]),
    num_batch=hp.choice("num_batch", [1, 2]),
    num_edges=hp.choice("num_edges", [1]),
    num_processes=hp.choice("num_processes", [16]),
    ppo_epoch=hp.choice("ppo_epoch", [3, 4, 10, 15]),
    tgt_success_rate=hp.choice("tgt_success_rate", [0.75, 0.8, 0.9]),
    train_steps=hp.choice("train_steps", [20, 25, 30]),
)


search_debug = copy.deepcopy(search_starcraft)
search_debug.update(
    kernel_size=1,
    stride=1,
)
debug_default = copy.deepcopy(starcraft_default)
debug_default.update(
    kernel_size=1,
    stride=1,
)
configs = dict(
    search_starcraft=search_starcraft,
    search_debug=search_debug,
    starcraft_default=starcraft_default,
    debug_default=debug_default,
    search=search,
)
