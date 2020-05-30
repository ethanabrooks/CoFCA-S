from rl_utils import hierarchical_parse_args

from ppo.arguments import build_parser
from ppo.train import Train
from ppo.water_maze import WaterMaze


def main(log_dir, seed, **kwargs):
    class _Train(Train):
        @staticmethod
        def make_env(seed, rank, env_id, add_timestep, **env_args):
            return WaterMaze(**env_args, seed=seed + rank)

    _Train().run(**kwargs, seed=seed, log_dir=log_dir)


def args():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--time-limit", type=int)
    WaterMaze.add_arguments(parsers.env)
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(args()))
