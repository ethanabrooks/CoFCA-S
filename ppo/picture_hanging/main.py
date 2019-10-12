from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.train
from ppo.picture_hanging.exp import Agent
from ppo.picture_hanging.env import Env
import ppo.picture_hanging.exp
import ppo.picture_hanging.baseline


def train(**_kwargs):
    class Train(ppo.train.Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            return Env(**env_args, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=False, **agent_args
        ):
            agent_args.update(
                action_space=envs.action_space, observation_space=envs.observation_space
            )
            if baseline:
                return ppo.picture_hanging.baseline.Agent(
                    entropy_coef=entropy_coef,
                    recurrence=ppo.picture_hanging.baseline.Recurrence(**agent_args),
                )
            else:
                return ppo.picture_hanging.exp.Agent(
                    entropy_coef=entropy_coef,
                    recurrence=(ppo.picture_hanging.exp.Recurrence(**agent_args)),
                )

        # def run_epoch(self, *args, **kwargs):
        #     dictionary = super().run_epoch(*args, **kwargs)
        #     rewards = dictionary["rewards"]
        #     if (
        #         increment_curriculum_at
        #         and rewards
        #         and sum(rewards) / len(rewards) > increment_curriculum_at
        #     ):
        #         self.envs.increment_curriculum()
        #     return dictionary

    Train(**_kwargs, time_limit=None).run()


def cli():
    parsers = ppo.arguments.build_parser()
    parsers.main.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parsers.main.add_argument("--eval-steps", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--bidirectional", action="store_true")
    parsers.agent.add_argument("--baseline", action="store_true")
    parsers.env.add_argument("--width", type=int, default=1)
    parsers.env.add_argument("--speed", type=float, default=0.1)
    parsers.env.add_argument("--n-train", type=int, default=3)
    parsers.env.add_argument("--n-eval", type=int, default=6)
    args = hierarchical_parse_args(parsers.main)
    train(**args)


if __name__ == "__main__":
    cli()
