from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args
from pathlib import Path

import ppo
import ppo.events.agent
from ppo.arguments import build_parser, get_args
from ppo.events.agent import Agent
from ppo.events.wrapper import Obs
from ppo.train import Train
import torch.nn as nn


def cli():
    Train(**get_args())


def exp_main(gridworld_args, wrapper_args, subtask, base, **kwargs):
    class _Train(Train):
        @staticmethod
        def make_env(time_limit, seed, rank, **kwargs):
            env = TimeLimit(
                max_episode_steps=time_limit,
                env=ppo.events.Wrapper(
                    **wrapper_args, env=ppo.events.Gridworld(**gridworld_args)
                ),
            )
            env.seed(seed + rank)
            return env

        def build_agent(self, envs, recurrent=None, device=None, **agent_args):
            return Agent(
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                **agent_args
            )

    _Train(**kwargs)


def exp_cli():
    parser = build_parser()
    parser.add_argument("--subtask")
    parser.add_argument("--base", action="store_true")
    gridworld_parser = parser.add_argument_group("gridworld_args")
    gridworld_parser.add_argument("--height", help="", type=int, default=4)
    gridworld_parser.add_argument("--width", help="", type=int, default=4)
    gridworld_parser.add_argument("--cook-time", help="", type=int, default=2)
    gridworld_parser.add_argument("--time-to-heat-oven", help="", type=int, default=3)
    gridworld_parser.add_argument("--doorbell-prob", help="", type=float, default=0.05)
    gridworld_parser.add_argument("--mouse-prob", help="", type=float, default=0.2)
    gridworld_parser.add_argument("--baby-prob", help="", type=float, default=0.1)
    gridworld_parser.add_argument("--mess-prob", help="", type=float, default=0.02)
    gridworld_parser.add_argument("--fly-prob", help="", type=float, default=0.005)
    wrapper_parser = parser.add_argument_group("wrapper_args")
    wrapper_parser.add_argument("--n-active-subtasks", help="", type=int, required=True)
    wrapper_parser.add_argument("--watch-baby-range", help="", type=int, default=2)
    wrapper_parser.add_argument("--avoid-dog-range", help="", type=int, default=2)
    wrapper_parser.add_argument("--door-time-limit", help="", type=int, default=10)
    wrapper_parser.add_argument("--max-time-outside", help="", type=int, default=15)
    exp_main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    exp_cli()
