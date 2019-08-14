import re
import time

import numpy as np
from pathlib import Path

from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter

import ppo
import ppo.events.agent
from ppo.arguments import build_parser, get_args
from ppo.events.agent import Agent
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu
import torch


def cli():
    Train(**get_args())


def exp_main(gridworld_args, wrapper_args, base, debug, **kwargs):
    class _Train(Train):
        def __init__(
            self, run_id, log_dir: Path, save_dir: Path, save_interval: int, **kwargs
        ):
            if log_dir:
                self.writer = SummaryWriter(logdir=str(log_dir))
            else:
                self.writer = None
            self.run_id = run_id
            self.save_dir = save_dir
            self.save_interval = save_interval
            self.setup(**kwargs)
            self.last_save = time.time()  # dummy save

        def _train(self):
            if (
                self.save_dir
                and self.save_interval
                and (time.time() - self.last_save >= self.save_interval)
            ):
                self._save(self.save_dir)
                self.last_save = time.time()
            return super()._train()

        def get_device(self):
            match = re.search("\d+$", self.run_id)
            if match:
                device_num = int(match.group()) % get_n_gpu()
            else:
                device_num = get_random_gpu()

            return torch.device("cuda", device_num)

        @staticmethod
        def make_env(time_limit, seed, rank, evaluation, **kwargs):
            env = ppo.events.Gridworld(**gridworld_args)
            if base:
                raise NotImplementedError
                env = ppo.events.BaseWrapper(
                    **wrapper_args, evaluation=evaluation, env=env
                )
            else:
                env = ppo.events.Wrapper(**wrapper_args, evaluation=evaluation, env=env)
            env = TimeLimit(max_episode_steps=time_limit, env=env)
            env.seed(seed + rank)
            return env

        def build_agent(self, envs, recurrent=None, device=None, **agent_args):
            if base:
                return super().build_agent(envs, recurrent=recurrent, **agent_args)
            return Agent(
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                debug=debug,
                **agent_args
            )

        def _log_result(self, result: dict):
            if self.writer is None:
                return
            total_num_steps = (self.i + 1) * self.processes * self.num_steps
            for k, v in result.items():
                self.writer.add_scalar(k, np.mean(v), total_num_steps)

    _Train(**kwargs).run()


def exp_cli():
    parser = build_parser()
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--debug", action="store_true")
    gridworld_parser = parser.add_argument_group("gridworld_args")
    gridworld_parser.add_argument("--height", help="", type=int, default=4)
    gridworld_parser.add_argument("--width", help="", type=int, default=4)
    gridworld_parser.add_argument("--cook-time", help="", type=int, default=2)
    gridworld_parser.add_argument("--time-to-heat-oven", help="", type=int, default=3)
    gridworld_parser.add_argument("--doorbell-prob", help="", type=float, default=0.05)
    gridworld_parser.add_argument("--mouse-prob", help="", type=float, default=0.2)
    gridworld_parser.add_argument("--baby-prob", help="", type=float, default=0.1)
    gridworld_parser.add_argument("--mess-prob", help="", type=float, default=0.01)
    gridworld_parser.add_argument("--fly-prob", help="", type=float, default=0.005)
    gridworld_parser.add_argument("--toward-cat-prob", help="", type=float, default=0.5)
    wrapper_parser = parser.add_argument_group("wrapper_args")
    wrapper_parser.add_argument("--n-active-subtasks", help="", type=int, required=True)
    wrapper_parser.add_argument("--watch-baby-range", help="", type=int, default=2)
    wrapper_parser.add_argument("--avoid-dog-range", help="", type=int, default=2)
    wrapper_parser.add_argument("--door-time-limit", help="", type=int, default=10)
    wrapper_parser.add_argument("--max-time-outside", help="", type=int, default=15)
    wrapper_parser.add_argument("--subtask", dest="subtasks", action="append")
    wrapper_parser.add_argument("--held-out", nargs="*", action="append", default=[])
    exp_main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    exp_cli()
