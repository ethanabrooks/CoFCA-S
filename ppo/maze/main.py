from rl_utils import hierarchical_parse_args

import ppo.maze
from ppo import maze, gntm
from ppo.main import build_parser
from ppo.train import Train


def train_maze(**kwargs):
    class TrainMaze(Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            assert time_limit
            return maze.Env(**env_args, time_limit=time_limit, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "one-shot":
                recurrence = ppo.maze.baselines.one_shot.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            else:
                assert baseline is None
                recurrence = maze.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainMaze(**kwargs).run()


def maze_cli():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--height", type=int, required=True)
    parsers.env.add_argument("--width", type=int, required=True)
    parsers.agent.add_argument("--baseline", choices=["one-shot"])
    train_maze(**hierarchical_parse_args(parser))
