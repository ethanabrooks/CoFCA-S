import itertools
import re
import time
from abc import ABC
from pathlib import Path

import ray
import torch
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter
import socket

import ppo.agent
import ppo.subtasks.agent
from ppo import subtasks
from ppo.arguments import build_parser
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu, k_scalar_pairs


def main(log_dir, baseline, seed, **kwargs):
    class _Train(Train):
        def __init__(
            self,
            run_id,
            log_dir: Path,
            save_interval: int,
            num_processes: int,
            num_steps: int,
            **kwargs,
        ):
            self.num_steps = num_steps
            self.num_processes = num_processes
            self.run_id = run_id
            self.save_interval = save_interval
            self.log_dir = log_dir
            if log_dir:
                self.writer = SummaryWriter(logdir=str(log_dir))
            else:
                self.writer = None
            self.setup(**kwargs, num_processes=num_processes, num_steps=num_steps)
            self.last_save = time.time()  # dummy save

        def build_agent(self, envs, debug=False, **agent_args):
            if baseline == "default":
                return ppo.agent.Agent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    **agent_args,
                )
            elif baseline == "oh-et-al":
                raise NotImplementedError
            return ppo.subtasks.agent.Agent(
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                baseline=baseline,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return subtasks.bandit.Env(**env_args, baseline=baseline, seed=seed + rank)

        def run(self):
            for _ in itertools.count():
                for result in self.make_train_iterator():
                    if self.writer is not None:
                        total_num_steps = (
                            (self.i + 1) * self.num_processes * self.num_steps
                        )
                        for k, v in k_scalar_pairs(**result):
                            self.writer.add_scalar(k, v, total_num_steps)

                    if (
                        self.log_dir
                        and self.save_interval
                        and (time.time() - self.last_save >= self.save_interval)
                    ):
                        self._save(str(self.log_dir))
                        self.last_save = time.time()

        def get_device(self):
            match = re.search("\d+$", self.run_id)
            if match:
                device_num = int(match.group()) % get_n_gpu()
            else:
                device_num = get_random_gpu()

            return torch.device("cuda", device_num)

    _Train(**kwargs, seed=seed, log_dir=log_dir).run()


def bandit_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--baseline", choices=["oh-et-al", "default"])
    parsers.env.add_argument("--n-lines", type=int, required=True)
    parsers.env.add_argument("--flip-prob", type=float, required=True)
    parsers.agent.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(bandit_args()))
