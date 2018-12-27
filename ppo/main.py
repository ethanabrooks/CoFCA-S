import argparse
from pathlib import Path

from hsr.util import env_wrapper, xml_setter
from torch import nn as nn
from utils.argparse import parse_vector

from ppo.train import train
from utils.argparse import parse_groups, parse_activation, parse_space


def cli():
    train(**parse_groups(build_parser()))


def hsr_cli():
    parser = get_hsr_parser()
    env_wrapper(train)(**parse_groups(parser))


def unsupervised_cli():
    parser = get_unsupervised_parser()
    args_dict = parse_groups(parser)
    args_dict.update(env_name='unsupervised')
    env_wrapper(train)(**args_dict)


def build_parser():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--normalize', action='store_true')
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
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
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
        '--env-name',
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
    parser.add_argument(
        '--add-timestep',
        action='store_true',
        default=False,
        help='add timestep to observations')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')

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
    return parser


def add_env_args(parser):
    parser.add_argument(
        '--image-dims', type=parse_vector(length=2, delim=','), default='800,800')
    parser.add_argument('--block-space', type=parse_space(dim=4))
    parser.add_argument('--min-lift-height', type=float, default=None)
    parser.add_argument('--no-random-reset', action='store_true')
    parser.add_argument('--obs-type', type=str, default=None)
    parser.add_argument('--randomize-pose', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-freq', type=int, default=None)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record-separate-episodes', action='store_true')
    parser.add_argument('--record-freq', type=int, default=None)
    parser.add_argument('--record-path', type=Path, default=None)
    parser.add_argument('--steps-per-action', type=int, required=True)


def add_wrapper_args(parser):
    parser.add_argument('--xml-file', type=Path, default='models/world.xml')
    parser.add_argument('--set-xml', type=xml_setter, action='append', nargs='*')
    parser.add_argument('--use-dof', type=str, action='append', default=[])
    parser.add_argument('--geofence', type=float, required=True)
    parser.add_argument('--n-blocks', type=int, required=True)
    parser.add_argument('--goal-space', type=parse_space(dim=3),
                        required=True)  # TODO


def get_hsr_parser():
    parser = build_parser()
    parser.add_argument('--max-steps', type=int)
    env_parser = parser.add_argument_group('env_args')
    add_env_args(env_parser)
    add_wrapper_args(parser.add_argument_group('wrapper_args'))
    return parser


def get_unsupervised_parser():
    parser = get_hsr_parser()
    unsupervised_parser = parser.add_argument_group('unsupervised_args')
    unsupervised_parser.add_argument(
        '--gan-hidden-size', type=int, default=256)
    unsupervised_parser.add_argument('--gan-num-layers', type=int, default=3)
    unsupervised_parser.add_argument(
        '--gan-activation', type=parse_activation, default=nn.ReLU())
    return parser