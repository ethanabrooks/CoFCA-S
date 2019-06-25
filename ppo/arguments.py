# stdlib
# third party
import argparse
from pathlib import Path

import torch.nn as nn

from rl_utils import hierarchical_parse_args


def parse_activation(string):
    return dict(relu=nn.ReLU)[string]


def build_parser():
    parser = argparse.ArgumentParser(
        description='RL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run-id', help=' ')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='discount factor for rewards')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--use-gae',
                        action='store_true',
                        default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau',
                        type=float,
                        default=0.95,
                        help='gae parameter')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--num-processes',
                        type=int,
                        default=16,
                        help='how many training CPU processes to use')
    parser.add_argument('--num-steps',
                        type=int,
                        default=5,
                        help='number of forward steps in A2C')
    parser.add_argument('--log-interval',
                        type=int,
                        default=10,
                        help='log interval, one log per n updates')
    parser.add_argument('--save-interval',
                        type=int,
                        default=100,
                        help='save interval, one save per n updates')
    parser.add_argument('--eval-interval',
                        type=int,
                        default=None,
                        help='eval interval, one eval per n updates')
    parser.add_argument('--env',
                        dest='env_id',
                        default='PongNoFrameskip-v4',
                        help='environment to train on')
    parser.add_argument('--load-path', type=Path)
    parser.add_argument('--log-dir',
                        type=Path,
                        help='directory to save agent logs')
    parser.add_argument('--no-cuda',
                        dest='cuda',
                        action='store_false',
                        help='enables CUDA training')
    parser.add_argument('--synchronous', action='store_true')
    parser.add_argument('--add-timestep',
                        action='store_true',
                        default=False,
                        help='add timestep to observations')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='number of batches for ppo')
    parser.add_argument('--success-reward', type=float)
    parser.add_argument('--target-success-rate', type=float)
    parser.add_argument('--max-episode-steps', type=int)

    agent_parser = parser.add_argument_group('agent_args')
    agent_parser.add_argument('--logic', action='store_true')
    agent_parser.add_argument('--recurrent', action='store_true')
    agent_parser.add_argument('--hidden-size', type=int, default=256)
    agent_parser.add_argument('--num-layers', type=int, default=3)
    agent_parser.add_argument('--activation',
                              type=parse_activation,
                              default=nn.ReLU())
    agent_parser.add_argument('--entropy-coef',
                              type=float,
                              default=0.01,
                              help='entropy term coefficient')

    ppo_parser = parser.add_argument_group('ppo_args')
    ppo_parser.add_argument('--clip-param',
                            type=float,
                            default=0.2,
                            help='ppo clip parameter')
    ppo_parser.add_argument('--ppo-epoch',
                            type=int,
                            default=4,
                            help='number of ppo epochs')
    ppo_parser.add_argument('--value-loss-coef',
                            type=float,
                            default=0.5,
                            help='value loss coefficient')
    ppo_parser.add_argument('--learning-rate',
                            type=float,
                            default=7e-4,
                            help='(default: 7e-4)')
    ppo_parser.add_argument('--eps',
                            type=float,
                            default=1e-5,
                            help='RMSprop optimizer epsilon')
    ppo_parser.add_argument('--max-grad-norm',
                            type=float,
                            default=0.5,
                            help='max norm of gradients')
    return parser


def get_args():
    return hierarchical_parse_args(build_parser())


# def get_hsr_args():
#     parser = build_parser()
#     env_parser = parser.add_argument_group('env_args')
#     add_env_args(env_parser)
#     add_wrapper_args(parser.add_argument_group('wrapper_args'))
#     return hierarchical_parse_args(parser)
