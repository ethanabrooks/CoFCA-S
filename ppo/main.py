import itertools
import re
import time
from pathlib import Path

import torch
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import bandit, gntm, maze, values, mdp
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu, k_scalar_pairs


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

    def run(self):
        for _ in itertools.count():
            for result in self.make_train_iterator():
                if self.writer is not None:
                    total_num_steps = (self.i + 1) * self.num_processes * self.num_steps
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


def train_bandit(**kwargs):
    class TrainBandit(_Train):
        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return bandit.bandit.Env(**env_args, seed=seed + rank)

        def build_agent(self, envs, entropy_coef=None, baseline=None, **agent_args):
            if baseline == "oh-et-al":
                recurrence = bandit.baselines.oh_et_al.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            else:
                assert baseline is None
                recurrence = bandit.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainBandit(**kwargs).run()


def train_maze(**kwargs):
    class TrainMaze(_Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            assert time_limit
            return maze.Env(**env_args, time_limit=time_limit, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "one-shot":
                recurrence = ppo.maze.baselines.one_shot.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            else:
                assert baseline is None
                recurrence = maze.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainMaze(**kwargs).run()


def train_values(time_limit, **kwargs):
    class TrainValues(_Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            assert time_limit
            return values.Env(**env_args, time_limit=time_limit, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "one-shot":
                raise NotImplementedError
            else:
                assert baseline is None
                recurrence = values.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    time_limit=time_limit,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainValues(time_limit=time_limit, **kwargs).run()


def train_mdp(**kwargs):
    class TrainBandit(_Train):
        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return mdp.Env(**env_args, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "oh-et-al":
                raise NotImplementedError
            else:
                assert baseline is None
                recurrence = mdp.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainBandit(**kwargs).run()


def build_parser():
    parsers = ppo.arguments.build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    return parsers


def bandit_cli():
    parsers = build_parser()
    parsers.env.add_argument("--n-lines", type=int, required=True)
    parsers.env.add_argument("--flip-prob", type=float, required=True)
    parsers.agent.add_argument("--baseline", choices=["oh-et-al"])
    train_bandit(**hierarchical_parse_args(parsers.main))


def maze_cli():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--height", type=int, required=True)
    parsers.env.add_argument("--width", type=int, required=True)
    parsers.agent.add_argument("--baseline", choices=["one-shot"])
    train_maze(**hierarchical_parse_args(parser))


def values_cli():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--size", type=int, required=True)
    parsers.env.add_argument("--min-reward", type=int, required=True)
    parsers.env.add_argument("--max-reward", type=int, required=True)
    parsers.agent.add_argument("--baseline", choices=["one-shot"])
    parsers.ppo.add_argument("--aux-loss-only", action="store_true")
    train_values(**hierarchical_parse_args(parser))


def mdp_cli():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--n-states", type=int, required=True)
    parsers.env.add_argument("--delayed-reward", action="store_true")
    parsers.agent.add_argument("--baseline", choices=["one-shot"])
    train_mdp(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    values_cli()
