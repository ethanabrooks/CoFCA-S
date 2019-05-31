# stdlib
import itertools
from pathlib import Path

import gym

# noinspection PyUnresolvedReferences
from gym.wrappers import TimeLimit

import gridworld_env
from gridworld_env import SubtasksGridWorld
from hsr.util import env_wrapper
from ppo.arguments import build_parser, get_args, get_hsr_args
from ppo.train import Trainer
from rl_utils import hierarchical_parse_args


def cli():
    Trainer(**get_args())


def hsr_cli():
    args = get_hsr_args()
    env_wrapper(Trainer)(**args)


def teacher_cli():
    parser = build_parser()
    parser.add_argument('--save-dir', type=Path)
    kwargs = hierarchical_parse_args(parser)
    env_args = gridworld_env.get_args(kwargs['env_id'])

    class Wrapper(gym.ObservationWrapper):
        def observation(self, observation):
            import ipdb; ipdb.set_trace()

    def train(env_id, task_types, task_counts, object_types, n_subtasks, **_kwargs):

        task_iter = itertools.product(task_types, task_counts, object_types,
                                      repeat=n_subtasks)
        for i, task in enumerate(task_iter):
            task = list(zip(*[iter(task)] * 3))
            log_dir = log_dir.joinpath(str(i))
            load_path = log_dir.joinpath('checkpoint.pt')
            if not load_path.exists():
                load_path = None

            class SubtasksTrainer(Trainer):
                @staticmethod
                def make_env(seed, rank, add_timestep, max_episode_steps=None,
                             **__kwargs):
                    env = Wrapper(SubtasksGridWorld(**__kwargs, task=task))
                    env.seed(seed + rank)
                    if max_episode_steps:
                        env = TimeLimit(env, max_episode_seconds=max_episode_steps)
                    return env

            SubtasksTrainer(log_dir=log_dir,
                            load_path=load_path,
                            env_id=env_id,
                            **_kwargs)

    train(**kwargs, **env_args)


if __name__ == "__main__":
    cli()
