from rl_utils import hierarchical_parse_args

from ppo import values
from ppo.main import build_parser
from ppo.train import Train


def train_values(time_limit, **kwargs):
    class TrainValues(Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            assert time_limit
            return values.Env(**env_args, time_limit=time_limit, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "one-shot":
                raise NotImplementedError
            else:
                assert baseline is None
                recurrence = values.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    time_limit=time_limit,
                    **agent_args,
                )
            return values.gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainValues(time_limit=time_limit, **kwargs).run()


def values_cli():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--size", type=int, required=True)
    parsers.env.add_argument("--min-reward", type=int, required=True)
    parsers.env.add_argument("--max-reward", type=int, required=True)
    parsers.agent.add_argument("--baseline", choices=["one-shot"])
    parsers.ppo.add_argument("--aux-loss-only", action="store_true")
    train_values(**hierarchical_parse_args(parser))
