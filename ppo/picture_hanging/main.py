from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.train
from ppo.picture_hanging.agent import Agent
from ppo.picture_hanging.env import Env
from ppo.picture_hanging.recurrence import Recurrence


def train(increment_curriculum_at, **_kwargs):
    class Train(ppo.train.Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            return Env(**env_args, seed=seed + rank)

        def build_agent(self, envs, recurrent=None, entropy_coef=None, **agent_args):
            recurrence = Recurrence(
                action_space=envs.action_space,
                observation_space=envs.observation_space,
                **agent_args,
            )
            return Agent(entropy_coef=entropy_coef, recurrence=recurrence)

        def run_epoch(self, *args, **kwargs):
            dictionary = super().run_epoch(*args, **kwargs)
            rewards = dictionary["rewards"]
            if (
                increment_curriculum_at
                and rewards
                and sum(rewards) / len(rewards) > increment_curriculum_at
            ):
                self.envs.increment_curriculum()
            return dictionary

    Train(**_kwargs, time_limit=None).run()


def cli():
    parsers = ppo.arguments.build_parser()
    parsers.main.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parsers.main.add_argument("--increment-curriculum-at", type=float)
    parsers.agent.add_argument("--debug", action="store_true")
    # parsers.agent.add_argument("--kernel-radius", type=int, default=5)
    parsers.env.add_argument("--single-step", action="store_true")
    parsers.env.add_argument("--width", type=int, default=100)
    parsers.env.add_argument("--min-pictures", type=int, default=2)
    parsers.env.add_argument("--max-pictures", type=int, default=20)
    args = hierarchical_parse_args(parsers.main)
    train(**args)


if __name__ == "__main__":
    cli()
