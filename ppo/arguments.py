# stdlib
# third party
import argparse
from pathlib import Path

from rl_utils import hierarchical_parse_args
import torch.nn as nn


ACTIVATIONS = dict(
    selu=nn.SELU(), prelu=nn.PReLU(), leaky=nn.LeakyReLU(), relu=nn.ReLU()
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="RL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-id", help=" ")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument("--tau", type=float, default=0.95, help="gae parameter")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument(
        "--num-processes",
        type=int,
        required=True,
        help="how many training CPU processes to use",
    )
    parser.add_argument(
        "--num-steps", type=int, required=True, help="number of forward steps in A2C"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="save interval, one save per n updates",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="eval interval, one eval per n updates",
    )
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--log-dir", type=Path, help="directory to save agent logs")
    parser.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="enables CUDA training"
    )
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument(
        "--num-batch", type=int, required=True, help="number of batches for ppo"
    )
    parser.add_argument("--success-reward", type=float)

    agent_parser = parser.add_argument_group("agent_args")
    agent_parser.add_argument("--recurrent", action="store_true")
    agent_parser.add_argument("--hidden-size", type=int, required=True)
    agent_parser.add_argument("--num-layers", type=int, required=True)
    agent_parser.add_argument(
        "--activation",
        type=lambda x: ACTIVATIONS[x],
        choices=ACTIVATIONS.values(),
        default=nn.ReLU(),
    )
    agent_parser.add_argument(
        "--entropy-coef", type=float, required=True, help="entropy term coefficient"
    )

    ppo_parser = parser.add_argument_group("ppo_args")
    ppo_parser.add_argument(
        "--clip-param", type=float, default=0.2, help="ppo clip parameter"
    )
    ppo_parser.add_argument(
        "--ppo-epoch", type=int, required=True, help="number of ppo epochs"
    )
    ppo_parser.add_argument(
        "--value-loss-coef", type=float, default=0.5, help="value loss coefficient"
    )
    ppo_parser.add_argument("--learning-rate", type=float, required=True, help="")
    ppo_parser.add_argument(
        "--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon"
    )
    ppo_parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="max norm of gradients"
    )
    env_parser = parser.add_argument_group("env_args")
    env_parser.add_argument(
        "--env",
        dest="env_id",
        default="PongNoFrameskip-v4",
        help="environment to train on",
    )
    env_parser.add_argument(
        "--add-timestep",
        action="store_true",
        default=False,
        help="add timestep to observations",
    )

    return parser


def get_args():
    return hierarchical_parse_args(build_parser())


# def get_hsr_args():
#     parser = build_parser()
#     env_parser = parser.add_argument_group('env_args')
#     add_env_args(env_parser)
#     add_wrapper_args(parser.add_argument_group('wrapper_args'))
#     return hierarchical_parse_args(parser)
def get_parser_with_exp_args():
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
    return parser
