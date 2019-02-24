import argparse
import functools
from pathlib import Path

import gym
from torch import nn as nn

import gridworld_env
import hsr.util
from ppo.env_adapter import (GridWorld, HSREnv, MoveGripperEnv, RandomGridWorld,
                             TasksGridWorld, TasksHSREnv,
                             TasksMoveGripperEnv)
from ppo.envs import wrap_env
from ppo.train import train
from utils import parse_groups

from ppo.update import SamplingStrategy
from ppo.util import parse_activation

try:
    import dm_control2gym
except ImportError:
    pass


def build_parser():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--no-task-in-obs', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.95,
        help='gae parameter (default: 0.95)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=30,
        help='log interval, one log per n seconds (default: 30 seconds)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=600,
        help='save interval, one save per n seconds (default: 10 minutes)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-frames',
        type=int,
        default=None,
        help='number of frames to train (default: None)')
    parser.add_argument(
        '--env-id',
        default='move-block',
        help='environment to train on (default: move-block)')
    parser.add_argument(
        '--log-dir',
        type=Path,
        help='directory to save agent logs and parameters')
    parser.add_argument(
        '--load-path',
        type=Path,
        help='directory to load agent parameters from')
    parser.add_argument(
        '--cuda', action='store_true', help='enables CUDA training')
    parser.add_argument('--synchronous', action='store_true')

    network_parser = parser.add_argument_group('network_args')
    network_parser.add_argument('--recurrent', action='store_true')
    network_parser.add_argument('--hidden-size', type=int, default=256)
    network_parser.add_argument('--num-layers', type=int, default=3)
    network_parser.add_argument(
        '--activation', type=parse_activation, default=nn.ReLU())

    ppo_parser = parser.add_argument_group('ppo_args')
    ppo_parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    ppo_parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    ppo_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    ppo_parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    ppo_parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    ppo_parser.add_argument(
        '--learning-rate', type=float, default=7e-4, help='(default: 7e-4)')
    ppo_parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    ppo_parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    ppo_parser.add_argument(
        '--temperature',
        type=float,
    )
    ppo_parser.add_argument(
        '--global-norm',
        action='store_true',
    )
    ppo_parser.add_argument(
        '--sampling-strategy',
        choices=[s.name for s in SamplingStrategy],
        default='experiment')
    return parser


def add_hsr_args(parser):
    parser.add_argument('--max-episode-steps', type=int)
    env_parser = parser.add_argument_group('env_args')
    hsr.util.add_env_args(env_parser)
    hsr.util.add_wrapper_args(parser.add_argument_group('wrapper_args'))


def add_tasks_args(parser):
    tasks_parser = parser.add_argument_group('tasks_args')
    tasks_parser.add_argument(
        '--gan-learning-rate',
        type=float,
        default=7e-4,
        help='(default: 7e-4)')
    tasks_parser.add_argument('--gan-num-samples', type=int)
    tasks_parser.add_argument('--gan-hidden-size', type=int)
    tasks_parser.add_argument('--gan-num-layers', type=int)
    tasks_parser.add_argument(
        '--gan-activation', type=parse_activation, default=nn.ReLU())
    tasks_parser.add_argument(
        '--gan-entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')


def cli():
    parser = build_parser()
    parser.add_argument('--max-episode-steps', type=int)
    parser.add_argument('--render', action='store_true')

    def make_gridworld_env_fn(env_id, max_episode_steps, eval,
                              **env_args):
        args = gridworld_env.get_args(env_id)
        if 'random' in args:
            class_ = RandomGridWorld
        else:
            class_ = GridWorld
        return functools.partial(
            wrap_env,
            env_thunk=lambda eval: class_(**env_args),
            max_episode_steps=max_episode_steps)

    def _train(env_id, max_episode_steps, no_task_in_obs, **kwargs):
        if 'GridWorld' in env_id:
            args = gridworld_env.get_args(env_id)
            if max_episode_steps is not None:
                args['max_episode_steps'] = max_episode_steps
            args.update(no_task_in_obs=no_task_in_obs)
            make_env = make_gridworld_env_fn(env_id, **args)

        else:

            def thunk(eval):
                if env_id.startswith("dm"):
                    _, domain, task = env_id.split('.')
                    return dm_control2gym.make(
                        domain_name=domain, task_name=task)
                else:
                    return gym.make(env_id)

            make_env = functools.partial(wrap_env, env_thunk=thunk)

        train(make_env=make_env, **kwargs)

    _train(**parse_groups(parser))


def tasks_cli():
    parser = build_parser()
    add_tasks_args(parser)
    parser.add_argument('--max-episode-steps', type=int)
    parser.add_argument('--render', action='store_true')

    def make_env_fn(max_episode_steps, **env_args):
        return functools.partial(
            wrap_env,
            env_thunk=lambda eval: TasksGridWorld(eval=eval, **env_args),
            max_episode_steps=max_episode_steps)

    def _train(env_id, max_episode_steps, no_task_in_obs, **kwargs):
        args = gridworld_env.get_args(env_id)
        args.update(no_task_in_obs=no_task_in_obs)
        if max_episode_steps is not None or 'max_episode_steps' not in args:
            args['max_episode_steps'] = max_episode_steps
        train(make_env=make_env_fn(**args), **kwargs)

    _train(**parse_groups(parser))


def hsr_cli():
    parser = build_parser()
    add_hsr_args(parser)

    def env_thunk(env_id, **kwargs):
        if env_id == 'move-gripper':
            return lambda eval: MoveGripperEnv(**kwargs)
        else:
            return lambda eval: HSREnv(**kwargs)

    def _train(env_id, env_args, max_episode_steps=None, **kwargs):
        make_env = functools.partial(
            wrap_env,
            env_thunk=env_thunk(env_id, **env_args),
            max_episode_steps=max_episode_steps)
        train(make_env=make_env, render=False, **kwargs)

    hsr.util.env_wrapper(_train)(**parse_groups(parser))


def tasks_hsr_cli():
    parser = build_parser()
    add_tasks_args(parser)
    add_hsr_args(parser)

    def env_thunk(env_id, eval, **env_args):
        if env_id == 'move-gripper':
            return lambda: TasksMoveGripperEnv(**env_args)
        else:
            return lambda: TasksHSREnv(**env_args)

    def _train(env_args, env_id, max_episode_steps, **kwargs):
        train(
            make_env=functools.partial(
                wrap_env,
                env_thunk=env_thunk(env_id, **env_args),
                max_episode_steps=max_episode_steps),
            **kwargs)

    hsr.util.env_wrapper(_train)(**parse_groups(parser), render=False)


if __name__ == "__main__":
    tasks_cli()
