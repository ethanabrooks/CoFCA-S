import argparse
import functools
from pathlib import Path
import pickle

import gym
from torch import nn as nn
from rl_utils import hierarchical_parse_args, parse_vector

import gridworld_env
import hsr.util
from ppo.env_adapter import AutoCurriculumHSREnv, HSREnv, SaveStateHSREnv, TasksGridWorld, TrainTasksGridWorld
from ppo.envs import wrap_env
from ppo.task_generator import SamplingStrategy
from ppo.train import train
from ppo.util import parse_activation

try:
    import dm_control2gym
except ImportError:
    pass


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99, help=' ')
    parser.add_argument(
        '--use-gae', action='store_true', default=False, help=' ')
    parser.add_argument('--deterministic-eval', action='store_true', help=' ')
    parser.add_argument('--tau', type=float, default=0.95, help=' ')
    parser.add_argument('--seed', type=int, default=1, help=' ')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-steps', type=int, default=5, help=' ')
    parser.add_argument('--log-interval', type=int, default=30, help=' ')
    parser.add_argument('--save-interval', type=int, default=None, help=' ')
    parser.add_argument('--eval-interval', type=int, default=None, help=' ')
    parser.add_argument('--num-frames', type=int, default=None, help=' ')
    parser.add_argument('--task-history', type=int, default=1000, help=' ')
    parser.add_argument(
        '--solved',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--num-solved',
        type=int,
        default=100,
    )
    parser.add_argument('--env-id', default='move-block', help=' ')
    parser.add_argument('--log-dir', type=Path, help=' ')
    parser.add_argument('--load-path', type=Path, help=' ')
    parser.add_argument(
        '--no-cuda', dest='cuda', action='store_false', help=' ')
    parser.add_argument('--synchronous', action='store_true')
    parser.add_argument('--num-processes', type=int, default=1)

    network_parser = parser.add_argument_group('network_args')
    network_parser.add_argument('--entropy-grade', type=float, default=10)
    network_parser.add_argument('--recurrent', action='store_true')
    network_parser.add_argument('--hidden-size', type=int, default=256)
    network_parser.add_argument('--num-layers', type=int, default=3)
    network_parser.add_argument(
        '--activation', type=parse_activation, default=nn.ReLU())

    ppo_parser = parser.add_argument_group('ppo_args')
    ppo_parser.add_argument('--clip-param', type=float, default=0.2, help=' ')
    ppo_parser.add_argument('--ppo-epoch', type=int, default=4, help=' ')
    ppo_parser.add_argument('--batch-size', type=int, default=32, help=' ')
    ppo_parser.add_argument(
        '--value-loss-coef', type=float, default=0.5, help=' ')
    ppo_parser.add_argument(
        '--entropy-coef', type=float, default=0.01, help=' ')
    ppo_parser.add_argument(
        '--learning-rate', type=float, default=7e-4, help=' ')
    ppo_parser.add_argument('--eps', type=float, default=1e-5, help=' ')
    ppo_parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help=' ')
    return parser


def add_hsr_args(parser):
    parser.add_argument('--max-episode-steps', type=int)
    env_parser = parser.add_argument_group('env_args')
    hsr.util.add_env_args(env_parser)
    hsr.util.add_wrapper_args(parser.add_argument_group('wrapper_args'))
    return env_parser


def add_tasks_args(parser):
    tasks_parser = parser.add_argument_group('tasks_args')
    tasks_parser.add_argument(
        '--temperature',
        type=float,
    )
    tasks_parser.add_argument(
        '--exploration-bonus',
        type=float,
    )
    tasks_parser.add_argument(
        '--sampling-strategy',
        choices=[s.name for s in SamplingStrategy] +
                ['reward-variance', 'reward-range'],
        default='experiment')
    gan_parser = parser.add_argument_group('gan_args')
    gan_parser.add_argument('--size-noise', type=int, default=4)
    gan_parser.add_argument('--gan-epoch', type=float, default=.9)


def cli():
    parser = build_parser()
    parser.add_argument('--max-episode-steps', type=int)

    def make_env_fn(max_episode_steps, **env_args):
        return functools.partial(
            wrap_env,
            env_thunk=lambda: TasksGridWorld(**env_args),
            max_episode_steps=max_episode_steps)

    def _train(env_id, max_episode_steps, **kwargs):
        if 'GridWorld' in env_id:
            args = gridworld_env.get_args(env_id)
            if max_episode_steps is not None:
                args['max_episode_steps'] = max_episode_steps
            make_env = make_env_fn(**args)

        else:

            def thunk():
                if env_id.startswith("dm"):
                    _, domain, task = env_id.split('.')
                    return dm_control2gym.make(
                        domain_name=domain, task_name=task)
                else:
                    return gym.make(env_id)

            make_env = functools.partial(wrap_env, env_thunk=thunk)

        train(make_env=make_env, **kwargs)

    _train(**hierarchical_parse_args(parser))


def tasks_cli():
    parser = build_parser()
    parser.add_argument('--render', action='store_true')
    add_tasks_args(parser)
    reward_based_task_parser = parser.add_argument_group(
        'reward_based_task_args')
    parser.add_argument('--max-episode-steps', type=int)
    reward_based_task_parser.add_argument(
        '--task-buffer-size', type=int, default=10, help=' ')
    reward_based_task_parser.add_argument(
        '--reward-bounds', type=parse_vector(2, ','), default=None, help=' ')

    def make_env_fn(max_episode_steps, **env_args):
        return functools.partial(
            wrap_env,
            env_thunk=lambda: TrainTasksGridWorld(**env_args),
            max_episode_steps=max_episode_steps)

    def _train(env_id, max_episode_steps, render, **kwargs):
        args = gridworld_env.get_args(env_id)
        args.update(max_episode_steps=max_episode_steps or args['max_episode_steps'],
                    render=render)
        train(make_env=make_env_fn(**args), **kwargs)

    _train(**hierarchical_parse_args(parser))


def hsr_cli():
    parser = build_parser()
    add_hsr_args(parser)

    def env_thunk(env_id, **kwargs):
        return lambda: HSREnv(**kwargs)

    def _train(env_id, env_args, max_episode_steps=None, **kwargs):
        make_env = functools.partial(
            wrap_env,
            env_thunk=env_thunk(env_id, **env_args),
            max_episode_steps=max_episode_steps)
        train(make_env=make_env, **kwargs)

    hsr.util.env_wrapper(_train)(**hierarchical_parse_args(parser))


def save_state_cli():
    parser = build_parser()
    env_parser = add_hsr_args(parser)
    env_parser.add_argument('--save-path', type=Path, required=True)

    def env_thunk(env_id, **env_args):
        return lambda: SaveStateHSREnv(**env_args)

    def _train(env_id, env_args, max_episode_steps=None, **kwargs):
        make_env = functools.partial(
            wrap_env,
            env_thunk=env_thunk(env_id, **env_args),
            max_episode_steps=max_episode_steps)
        train(make_env=make_env, **kwargs)

    hsr.util.env_wrapper(_train)(**hierarchical_parse_args(parser))


def unpickle(path: str):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def tasks_hsr_cli():
    parser = build_parser()
    add_tasks_args(parser)
    env_parser = add_hsr_args(parser)
    env_parser.add_argument('--start-states', type=unpickle, required=True)
    env_parser.add_argument(
        '--random-initial-steps',
        type=int,
        default=0,
        help='Environment steps on randomly sampled action at start of episode'
    )

    def env_thunk(env_id, start_states, **env_args):
        return lambda: AutoCurriculumHSREnv(*start_states, **env_args)

    def _train(env_args, env_id, max_episode_steps, **kwargs):
        train(
            make_env=functools.partial(
                wrap_env,
                env_thunk=env_thunk(env_id, **env_args),
                max_episode_steps=max_episode_steps),
            **kwargs)

    hsr.util.env_wrapper(_train)(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    tasks_cli()
