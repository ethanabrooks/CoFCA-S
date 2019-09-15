from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import blocks_world, gntm
from ppo.train import Train
from gym.wrappers import TimeLimit
import numpy as np


def build_parser():
    parsers = ppo.arguments.build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    return parsers


def train_blocks_world(increment_curriculum_at_n_satisfied, **kwargs):
    class TrainValues(Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            return blocks_world.Env(**env_args, seed=seed + rank)

        def run_epoch(self, *args, **kwargs):
            dictionary = super().run_epoch(*args, **kwargs)
            try:
                increment_curriculum = (
                    np.mean(dictionary["n_satisfied"])
                    > increment_curriculum_at_n_satisfied
                )
            except (TypeError, KeyError):
                increment_curriculum = False
            if increment_curriculum:
                self.envs.increment_curriculum()
            return dictionary

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "one-shot":
                raise NotImplementedError
            else:
                assert baseline is None
                recurrence = blocks_world.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )

            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainValues(**kwargs).run()


def blocks_world_cli():
    parsers = build_parser()
    parsers.env.add_argument("--n-cols", type=int, required=True)
    parsers.env.add_argument("--curriculum-level", type=int, default=0)
    parsers.main.add_argument("--increment-curriculum-at-n-satisfied", type=float)
    parsers.agent.add_argument("--num-slots", type=int, required=True)
    parsers.agent.add_argument("--slot-size", type=int, required=True)
    parsers.agent.add_argument("--embedding-size", type=int, required=True)
    parsers.agent.add_argument("--num-heads", type=int, required=True)
    train_blocks_world(**hierarchical_parse_args(parsers.main))


if __name__ == "__main__":
    blocks_world_cli()
