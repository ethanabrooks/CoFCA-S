import json
from collections import namedtuple
from pathlib import Path

import yaml

Parsers = namedtuple("Parser", "main agent ppo rollouts")


def load_config(yml_or_path: str):
    path = Path(yml_or_path)
    if path.exists():
        with path.open() as f:

            def parse():
                for k, v in yaml.load(f, Loader=yaml.FullLoader).items():
                    try:
                        yield k, v["value"]
                    except TypeError:
                        pass

            return dict(parse())
    return yaml.load(yml_or_path)


def add_arguments(parser):
    parser.add_argument(
        "--config",
        type=load_config,
        default={},
        help="Config that is used to update given params",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_true",
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--env", dest="env_id", help="environment to train on", default="CartPole-v0"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        help="eval interval, one eval per n updates",
        default=int(1e6),
    )
    parser.add_argument(
        "--eval_steps", type=int, help="number of steps for evaluation", default=500
    )
    parser.add_argument("--group")
    parser.add_argument("--load_path", type=Path)
    parser.add_argument(
        "--log_interval",
        type=int,
        help="log interval, one log per n updates",
        default=int(2e4),
    )
    parser.add_argument(
        "--no_cuda", dest="cuda", action="store_false", help="enables CUDA training"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Do not use wandb.")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--num_batch", type=int, help="number of batches for ppo", default=2
    )
    parser.add_argument("--num_frames", type=int, default=int(1e7))
    parser.add_argument(
        "--num_processes",
        type=int,
        help="how many training CPU processes to use",
        default=100,
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_eval", action="store_true")
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument(
        "--train_steps",
        type=int,
        help="number of forward steps in A2C",
        default=25,
    )
    # parser.add_argument("--success-reward", type=float)

    agent_parser = parser.add_argument_group("agent_args")
    agent_parser.add_argument(
        "--entropy_coef", type=float, help="entropy term coefficient", default=0.25
    )
    agent_parser.add_argument(
        "--hidden_size",
        type=int,
        default=150,
    )
    agent_parser.add_argument("--num_layers", type=int, default=1)
    agent_parser.add_argument("--recurrent", action="store_true")

    ppo_parser = parser.add_argument_group("ppo_args")
    ppo_parser.add_argument(
        "--clip_param", type=float, help="ppo clip parameter", default=0.2
    )
    ppo_parser.add_argument(
        "--ppo_epoch", type=int, help="number of ppo epochs", default=5
    )
    ppo_parser.add_argument(
        "--value_loss_coef", type=float, help="value loss coefficient", default=0.5
    )
    ppo_parser.add_argument("--learning_rate", type=float, help="", default=0.0025)
    ppo_parser.add_argument(
        "--eps", type=float, help="RMSprop optimizer epsilon", default=1e-5
    )
    ppo_parser.add_argument(
        "--max_grad_norm", type=float, help="max norm of gradients", default=0.5
    )
    rollouts_parser = parser.add_argument_group("rollouts_args")
    rollouts_parser.add_argument(
        "--gamma", type=float, help="discount factor for rewards", default=0.99
    )
    parser.add_argument(
        "--save_interval", type=int, help="how often to save.", default=int(2e4)
    )
    rollouts_parser.add_argument(
        "--tau", type=float, help="gae parameter", default=0.95
    )
    rollouts_parser.add_argument(
        "--use_gae",
        type=bool,
        help="use generalized advantage estimation",
    )

    return Parsers(
        main=parser, rollouts=rollouts_parser, ppo=ppo_parser, agent=agent_parser
    )
