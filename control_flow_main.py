from argparse import ArgumentParser
from pathlib import Path

from gym import spaces

import control_flow_agent
import debug_env
import env
import networks
from env import Action
from main import add_arguments
from trainer import Trainer
from utils import hierarchical_parse_args


def main(
    env_args,
    env_id,
    log_dir,
    lower_level,
    lower_level_load_path,
    max_eval_lines,
    min_eval_lines,
    render,
    seed,
    **kwargs,
):
    if lower_level_load_path:
        lower_level = "pre-trained"

    class _Trainer(Trainer):
        def build_agent(self, envs, debug=False, **agent_args):
            obs_space = envs.observation_space
            ll_action_space = spaces.Discrete(Action(*envs.action_space.nvec).lower)
            if lower_level == "train-alone":
                return networks.Agent(
                    lower_level=True,
                    obs_spaces=obs_space,
                    action_space=ll_action_space,
                    **agent_args,
                )
            agent_args.update(log_dir=log_dir)
            del agent_args["recurrent"]
            del agent_args["num_conv_layers"]
            return control_flow_agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                eval_lines=max_eval_lines,
                debug=render and debug,
                lower_level=lower_level,
                lower_level_load_path=lower_level_load_path,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluation, env_id=None):
            args = dict(
                **env_args,
                min_eval_lines=min_eval_lines,
                max_eval_lines=max_eval_lines,
                seed=seed + rank,
                rank=rank,
            )
            args["lower_level"] = lower_level
            args["break_on_fail"] = args["break_on_fail"] and render
            if not lower_level:
                args.update(world_size=1)
                return debug_env.Env(**args)
            else:
                return env.Env(**args)

        def process_infos(self, episode_counter, done, infos, **act_log):
            for d in infos:
                for k, v in d.items():
                    if k.startswith("cumulative_reward"):
                        episode_counter[k].append(v)
            super().process_infos(episode_counter, done, infos, **act_log)

    _Trainer.main(
        **kwargs, seed=seed, log_dir=log_dir, render=render, env_id="control-flow"
    )


def control_flow_args(parser):
    parsers = add_arguments(parser)
    parser = parsers.main
    parser.add_argument("--min-eval-lines", type=int, required=True)
    parser.add_argument("--max-eval-lines", type=int, required=True)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--lower-level", choices=["train-alone", "train-with-upper"])
    parser.add_argument("--lower-level-load-path")
    env.build_parser(parser.add_argument_group("env_args"))
    parsers.agent.add_argument("--lower-level-config", type=Path)
    parsers.agent.add_argument("--no-debug", dest="debug", action="store_false")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--no-pointer", action="store_true")
    parsers.agent.add_argument("--olsk", action="store_true")
    parsers.agent.add_argument("--transformer", action="store_true")
    parsers.agent.add_argument("--fuzz", action="store_true")
    parsers.agent.add_argument("--conv-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--task-embed-size", type=int, required=True)
    parsers.agent.add_argument("--lower-embed-size", type=int, required=True)
    parsers.agent.add_argument("--inventory-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--num-conv-layers", type=int, required=True)
    parsers.agent.add_argument("--num-edges", type=int, required=True)
    parsers.agent.add_argument("--gate-coef", type=float, required=True)
    parsers.agent.add_argument("--no-op-coef", type=float, required=True)
    parsers.agent.add_argument("--kernel-size", type=int, required=True)
    parsers.agent.add_argument("--stride", type=int, required=True)
    return parser


if __name__ == "__main__":
    PARSER = ArgumentParser()
    main(**hierarchical_parse_args(control_flow_args(PARSER)))
