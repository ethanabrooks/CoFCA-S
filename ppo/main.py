import re
import time

from pathlib import Path

from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

import ppo
import ppo.events.agent
from ppo.arguments import get_args, get_parser_with_exp_args
from ppo.events.agent import Agent
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu
import torch


def cli():
    Train(**get_args())


def exp_main(gridworld_args, wrapper_args, base, debug, **kwargs):
    class _Train(Train):
        def __init__(self, run_id, log_dir: Path, save_interval: int, **kwargs):
            self.run_id = run_id
            self.save_interval = save_interval
            self.logdir = str(log_dir)
            self.setup(**kwargs)
            self.last_save = time.time()  # dummy save

        def _train(self):
            if (
                self.logdir
                and self.save_interval
                and (time.time() - self.last_save >= self.save_interval)
            ):
                self._save(self.logdir)
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

    _Train(**kwargs).run()


def exp_cli():
    exp_main(**hierarchical_parse_args(get_parser_with_exp_args()))


if __name__ == "__main__":
    exp_cli()
