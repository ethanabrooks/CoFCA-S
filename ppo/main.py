# stdlib
import csv
import subprocess
from io import StringIO

import numpy as np

# noinspection PyUnresolvedReferences
# first party
import gridworld_env
from ppo.arguments import build_parser, get_args
from ppo.train import Trainer
from rl_utils import hierarchical_parse_args


def get_freer_gpu():
    nvidia_smi = subprocess.check_output(
        'nvidia-smi --format=csv --query-gpu=memory.free'.split(),
        universal_newlines=True)
    free_memory = [
        float(x[0].split()[0])
        for i, x in enumerate(csv.reader(StringIO(nvidia_smi))) if i > 0
    ]
    return np.argmax(free_memory)


def cli():
    Trainer().train(**get_args())


def logic_cli():
    parser = build_parser()
    env_args = parser.add_argument_group('env_args')
    env_args.add_argument('--partial', action='store_true')

    def _main(env_id, env_args, network_args, **kwargs):
        env_args.update(gridworld_env.get_args(env_id))
        del env_args['env_id']
        del env_args['class']
        network_args.update(logic=True)
        Trainer().train(
            env_id=env_id,
            env_args=env_args,
            network_args=network_args,
            **kwargs)

    _main(**hierarchical_parse_args(parser))


# def hsr_cli():
# args = get_hsr_args()
# env_wrapper(main)(**args)

if __name__ == "__main__":
    cli()
