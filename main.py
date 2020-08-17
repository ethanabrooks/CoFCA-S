import argparse
import json
from collections import namedtuple
from pathlib import Path

from torch import nn as nn

from configs import configs
from trainer import Trainer
from utils import hierarchical_parse_args

ACTIVATIONS = dict(
    selu=nn.SELU(), prelu=nn.PReLU(), leaky=nn.LeakyReLU(), relu=nn.ReLU()
)
Parsers = namedtuple("Parser", "main agent ppo rollouts")


def get_config(name):
    if name is None:
        return {}
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return configs[name]


def add_arguments(parser):
    parser.add_argument("--config", type=get_config)
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--eval-interval", type=int, help="eval interval, one eval per n updates"
    )
    parser.add_argument("--eval-steps", type=int, help="number of steps for evaluation")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates",
    )
    parser.add_argument("--name")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--gpus-per-trial", "-g", type=int, default=1)
    parser.add_argument("--cpus-per-trial", "-c", type=int, default=6)
    parser.add_argument(
        "--num-epochs", type=int, help="number of updates to perform", required=True
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="how many training CPU processes to use",
        required=True,
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        help="Number of times to sample from the hyperparameter space. See tune docs for details: "
        "https://docs.ray.io/en/latest/tune/api_docs/execution.html?highlight=run#ray.tune.run",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--train-steps", type=int, help="number of forward steps in A2C", required=True
    )
    parser.add_argument(
        "--env",
        dest="env_id",
        default="PongNoFrameskip-v4",
        help="environment to train on",
    )
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--log-dir", type=Path, help="directory to save agent logs")
    parser.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="enables CUDA training"
    )
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument(
        "--num-batch", type=int, help="number of batches for ppo", required=True
    )
    # parser.add_argument("--success-reward", type=float)

    agent_parser = parser.add_argument_group("agent_args")
    agent_parser.add_argument(
        "--activation", type=lambda s: eval(f"nn.{s}"), default=nn.ReLU()
    )
    agent_parser.add_argument(
        "--entropy-coef", type=float, help="entropy term coefficient", required=True
    )
    agent_parser.add_argument("--hidden-size", type=int, required=True)
    agent_parser.add_argument("--num-layers", type=int)
    agent_parser.add_argument("--recurrent", action="store_true")

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
    ppo_parser.add_argument("--learning-rate", type=float, help="", required=True)
    ppo_parser.add_argument(
        "--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon"
    )
    ppo_parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="max norm of gradients"
    )
    rollouts_parser = parser.add_argument_group("rollouts_args")
    rollouts_parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument("--save-interval", type=int, help="how often to save.")
    rollouts_parser.add_argument(
        "--tau", type=float, default=0.95, help="gae parameter"
    )
    rollouts_parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )

    return Parsers(
        main=parser, rollouts=rollouts_parser, ppo=ppo_parser, agent=agent_parser
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    args = hierarchical_parse_args(PARSER)
    Trainer.main(**args)
