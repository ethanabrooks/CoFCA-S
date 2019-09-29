import numpy as np
from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import gntm, oh_et_al
from ppo.agent import Agent
from ppo.blocks_world import dnc, planner
from ppo.train import Train


def build_parser():
    parsers = ppo.arguments.build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int, required=True)
    parsers.agent.add_argument("--debug", action="store_true")
    return parsers


def train_oh_et_al(**_kwargs):
    class TrainOhEtAl(Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            return TimeLimit(
                oh_et_al.Env(**env_args, seed=seed + rank), max_episode_steps=time_limit
            )

        def build_agent(self, envs, recurrent=None, entropy_coef=None, **agent_args):
            recurrence = oh_et_al.Recurrence(
                action_space=envs.action_space,
                observation_space=envs.observation_space,
                **agent_args,
            )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainOhEtAl(**_kwargs).run()


def oh_et_al_cli():
    parsers = build_parser()
    parsers.env.add_argument("--height", type=int, default=3)
    parsers.env.add_argument("--width", type=int, default=3)
    parsers.env.add_argument("--n-subtasks", type=int, default=3)
    parsers.env.add_argument("--n-objects", type=int, default=3)
    parsers.env.add_argument("--implement-lower-level", action="store_true")
    train_oh_et_al(**hierarchical_parse_args(parsers.main))


if __name__ == "__main__":
    oh_et_al_cli()
