from rl_utils import hierarchical_parse_args

from ppo.arguments import build_parser
from ppo.subtasks.repeat_copy import RepeatCopy
from ppo.train import Train

ENV = RepeatCopy


def main(log_dir, seed, **kwargs):
    class _Train(Train):
        @staticmethod
        def make_env(seed, rank, env_id, add_timestep, **env_args):
            return ENV(**env_args, seed=seed + rank)

    _Train().run(**kwargs, seed=seed, log_dir=log_dir)


def args():
    parsers = build_parser()
    ENV.add_arguments(parsers.env)
    return parsers.main


if __name__ == "__main__":
    main(**hierarchical_parse_args(args()))
