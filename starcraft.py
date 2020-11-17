import inspect
from argparse import ArgumentParser
from pathlib import Path

from gym import spaces

import our_agent
import debug_env
import env
import agents
import ours
from env import Action
from lower_level import LowerLevel
from arguments import add_arguments
from trainer import Trainer


def main(**kwargs):
    class ControlFlowTrainer(Trainer):
        def build_agent(self, envs, debug=False, **agent_args):
            obs_space = envs.observation_space
            del agent_args["recurrent"]
            del agent_args["num_layers"]
            return our_agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                debug=debug,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluating, lower_level=None, env_id=None, **kwargs):
            kwargs.update(
                seed=seed + rank,
                rank=rank,
                lower_level=lower_level,
                evaluating=evaluating,
            )
            if not lower_level:
                kwargs.update(world_size=1)
                return debug_env.Env(**kwargs)
            return env.Env(**kwargs)

        @classmethod
        def structure_config(cls, **config):
            config = super().structure_config(**config)
            agent_args = config.pop("agent_args")
            env_args = {}
            gen_args = {}

            if config["lower_level_load_path"]:
                config["lower_level"] = "pre-trained"

            agent_args["eval_lines"] = config["max_eval_lines"]
            agent_args["debug"] = config["render"] or config["render_eval"]

            for k, v in config.items():
                if (
                    k in inspect.signature(env.Env.__init__).parameters
                    or k in inspect.signature(cls.make_env).parameters
                ):
                    if k == "lower_level":
                        if v:
                            print("lower_level specified. Using gridworld env")
                        else:
                            print("lower_level not specified. Using debug_env")
                    env_args[k] = v
                if k in inspect.signature(ours.Recurrence.__init__).parameters:
                    agent_args[k] = v
                if k in inspect.signature(our_agent.Agent.__init__).parameters:
                    agent_args[k] = v
                if k in inspect.signature(our_agent.Agent.__init__).parameters:
                    agent_args[k] = v
                if k in inspect.signature(cls.run).parameters:
                    gen_args[k] = v
            d = dict(env_args=env_args, agent_args=agent_args, **gen_args)
            return d

    kwargs.update(env_id="control-flow")
    ControlFlowTrainer.main(**kwargs)


def control_flow_args(parser):
    parsers = add_arguments(parser)
    parser = parsers.main
    parser.add_argument("--min-eval-lines", type=int)
    parser.add_argument("--max-eval-lines", type=int)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--lead", action="store_true")
    parser.add_argument("--lower-level", choices=["train-alone", "train-with-upper"])
    parser.add_argument("--lower-level-load-path")
    env.add_arguments(parser.add_argument_group("env_args"))
    parsers.agent.add_argument("--lower-level-config", type=Path)
    parsers.agent.add_argument("--no-debug", dest="debug", action="store_false")
    parsers.agent.add_argument("--debug-obs", action="store_true")
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
