from rl_utils import hierarchical_parse_args

import ppo.agent
import ppo.arguments
import ppo.picture_hanging.dnc
import ppo.picture_hanging.last_step
import ppo.picture_hanging.multi_step
import ppo.train
from ppo.picture_hanging.env import Env
from ppo.picture_hanging.multi_step import Agent


def train(agent, **_kwargs):
    class Train(ppo.train.Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            return Env(
                **env_args,
                seed=seed + rank,
                time_limit=time_limit,
                include_sizes=agent not in ["default", "dnc"],
            )

        def build_agent(
            self,
            envs,
            recurrent=None,
            entropy_coef=None,
            r_to_actor=None,
            debug=None,
            dnc_args=None,
            **agent_args,
        ):
            if agent == "simple":
                return ppo.agent.Agent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    entropy_coef=entropy_coef,
                    recurrent=recurrent,
                    **agent_args,
                )
            elif agent == "last-step":
                return ppo.picture_hanging.last_step.Agent(
                    entropy_coef=entropy_coef,
                    recurrence=ppo.picture_hanging.last_step.Recurrence(
                        **agent_args,
                        debug=debug,
                        action_space=envs.action_space,
                        observation_space=envs.observation_space,
                    ),
                )
            elif agent == "dnc":
                return ppo.picture_hanging.dnc.Agent(
                    entropy_coef=entropy_coef,
                    recurrence=ppo.picture_hanging.dnc.Recurrence(
                        **agent_args,
                        **dnc_args,
                        debug=debug,
                        action_space=envs.action_space,
                        observation_space=envs.observation_space,
                    ),
                )
            elif agent == "multi-step":
                return ppo.picture_hanging.multi_step.Agent(
                    entropy_coef=entropy_coef,
                    recurrence=(
                        ppo.picture_hanging.multi_step.Recurrence(
                            **agent_args,
                            debug=debug,
                            r_to_actor=r_to_actor,
                            action_space=envs.action_space,
                            observation_space=envs.observation_space,
                        )
                    ),
                )
            else:
                raise RuntimeError

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

    Train(**_kwargs).run()


def cli():
    parsers = ppo.arguments.build_parser()
    parsers.main.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parsers.main.add_argument("--eval-steps", type=int)
    parsers.main.add_argument("--time-limit", type=int, required=True)
    parsers.main.add_argument("--agent")
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--r-to-actor", action="store_true")
    parsers.env.add_argument("--width", type=int, default=100)
    parsers.env.add_argument("--speed", type=int, default=20)
    parsers.env.add_argument("--min-train", type=int, default=1)
    parsers.env.add_argument("--max-train", type=int, default=3)
    parsers.env.add_argument("--n-eval", type=int, default=6)
    dnc_parser = parsers.agent.add_argument_group("dnc_args")
    dnc_parser.add_argument("--num-slots", type=int, default=8)
    dnc_parser.add_argument("--slot-size", type=int, default=128)
    dnc_parser.add_argument("--num-heads", type=int, default=2)
    args = hierarchical_parse_args(parsers.main)
    train(**args)


if __name__ == "__main__":
    cli()
