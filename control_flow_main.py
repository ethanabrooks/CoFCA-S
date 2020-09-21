import inspect
from argparse import ArgumentParser
from pathlib import Path

import torch
from gym import spaces

import control_flow_agent
import debug_env
import env
import lines
import networks
import ours
from env import Action
from lower_level import LowerLevel
from main import add_arguments
from trainer import Trainer


def main(**kwargs):
    class ControlFlowTrainer(Trainer):
        def build_agent(
            self, envs, lower_level="train-alone", debug=False, **agent_args
        ):
            obs_space = envs.observation_space
            ll_action_space = spaces.Discrete(Action(*envs.action_space.nvec).lower)
            if lower_level == "train-alone":
                return networks.Agent(
                    lower_level=True,
                    obs_spaces=obs_space,
                    action_space=ll_action_space,
                    **agent_args,
                )
            del agent_args["recurrent"]
            del agent_args["num_layers"]
            return control_flow_agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                lower_level=lower_level,
                debug=debug,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluation, lower_level, env_id=None, **kwargs):
            kwargs.update(seed=seed + rank, rank=rank, lower_level=lower_level)
            if not lower_level:
                kwargs.update(world_size=1)
                return debug_env.Env(**kwargs)
            return env.Env(**kwargs)

        def structure_config(self, **config):
            config = super().structure_config(**config)
            agent_args = config.pop("agent_args")
            env_args = {}
            gen_args = {}

            if config["lower_level_load_path"]:
                config["lower_level"] = "pre-trained"

            agent_args["eval_lines"] = config["max_eval_lines"]
            agent_args["debug"] = config["render"] and config["debug"]

            for k, v in config.items():
                if (
                    k in inspect.signature(env.Env.__init__).parameters
                    or k in inspect.signature(self.make_env).parameters
                ):
                    env_args[k] = v
                if k in inspect.signature(ours.Recurrence.__init__).parameters:
                    agent_args[k] = v
                if k in inspect.signature(control_flow_agent.Agent.__init__).parameters:
                    agent_args[k] = v
                if k in inspect.signature(control_flow_agent.Agent.__init__).parameters:
                    agent_args[k] = v
                if k in inspect.signature(self.gen).parameters:
                    gen_args[k] = v
            d = dict(env_args=env_args, agent_args=agent_args, **gen_args)
            d = {
                "agent_args": {
                    "activation": torch.nn.ReLU(),
                    "conv_hidden_size": 64,
                    "debug": False,
                    "entropy_coef": 0.015,
                    "eval_lines": 50,
                    "fuzz": False,
                    "gate_coef": 0.01,
                    "hidden_size": 256,
                    "inventory_hidden_size": 128,
                    "kernel_size": 2,
                    "lower_embed_size": 64,
                    "lower_level": "pre-trained",
                    "lower_level_config": Path("checkpoint/lower.json"),
                    "lower_level_load_path": "checkpoint/lower.pt",
                    "no_op_coef": 0.0,
                    "no_pointer": False,
                    "no_roll": False,
                    "no_scan": False,
                    "num_edges": 2,
                    "num_layers": 0,
                    "olsk": False,
                    "recurrent": False,
                    "stride": 1,
                    "task_embed_size": 64,
                    "transformer": False,
                },
                "cuda": True,
                "cuda_deterministic": False,
                "env_args": {
                    "break_on_fail": False,
                    "control_flow_types": [
                        lines.Subtask,
                        lines.If,
                        lines.While,
                        lines.Else,
                    ],
                    "env_id": "control-flow",
                    "eval_condition_size": False,
                    "failure_buffer_size": 500,
                    "long_jump": False,
                    "lower_level": "pre-trained",
                    "max_eval_lines": 50,
                    "max_failure_sample_prob": 0.3,
                    "max_lines": 10,
                    "max_loops": 3,
                    "max_nesting_depth": 1,
                    "max_while_loops": 10,
                    "max_world_resamples": 50,
                    "min_eval_lines": 1,
                    "min_lines": 1,
                    "no_op_limit": 30,
                    "one_condition": False,
                    "reject_while_prob": 0.6,
                    "seed": 0,
                    "single_control_flow_type": False,
                    "subtasks_only": False,
                    "term_on": ["mine", "sell"],
                    "time_to_waste": 0,
                    "use_water": True,
                    "world_size": 6,
                },
                "env_id": "control-flow",
                "eval_interval": 100,
                "eval_steps": 500,
                "load_path": None,
                "log_interval": 10,
                "no_eval": False,
                "normalize": False,
                "num_batch": 1,
                "num_processes": 150,
                "ppo_args": {
                    "clip_param": 0.2,
                    "eps": 1e-05,
                    "learning_rate": 0.003,
                    "max_grad_norm": 0.5,
                    "num_batch": 1,
                    "ppo_epoch": 2,
                    "value_loss_coef": 0.5,
                },
                "render": False,
                "render_eval": False,
                "rollouts_args": {
                    "gamma": 0.99,
                    "num_processes": 150,
                    "tau": 0.95,
                    "use_gae": False,
                },
                "seed": 0,
                "synchronous": False,
                "train_steps": 25,
            }
            return d

    kwargs.update(env_id="control-flow")
    ControlFlowTrainer.main(**kwargs)


def control_flow_args(parser):
    parsers = add_arguments(parser)
    parser = parsers.main
    parser.add_argument("--min-eval-lines", type=int)
    parser.add_argument("--max-eval-lines", type=int)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--lower-level", choices=["train-alone", "train-with-upper"])
    parser.add_argument("--lower-level-load-path")
    env.add_arguments(parser.add_argument_group("env_args"))
    parsers.agent.add_argument("--lower-level-config", type=Path)
    parsers.agent.add_argument("--no-debug", dest="debug", action="store_false")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--no-pointer", action="store_true")
    parsers.agent.add_argument("--olsk", action="store_true")
    parsers.agent.add_argument("--transformer", action="store_true")
    parsers.agent.add_argument("--fuzz", action="store_true")
    parsers.agent.add_argument("--conv-hidden-size", type=int)
    parsers.agent.add_argument("--task-embed-size", type=int)
    parsers.agent.add_argument("--lower-embed-size", type=int)
    parsers.agent.add_argument("--inventory-hidden-size", type=int)
    parsers.agent.add_argument("--num-edges", type=int)
    parsers.agent.add_argument("--gate-coef", type=float)
    parsers.agent.add_argument("--no-op-coef", type=float)
    parsers.agent.add_argument("--kernel-size", type=int)
    parsers.agent.add_argument("--stride", type=int)
    return parser


if __name__ == "__main__":
    PARSER = ArgumentParser()
    main(**vars(control_flow_args(PARSER).parse_args()))
