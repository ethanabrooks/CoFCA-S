from rl_utils import hierarchical_parse_args

from ppo import mdp, gntm
from ppo.main import build_parser
from ppo.train import Train


def train_mdp(**kwargs):
    class TrainBandit(Train):
        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return mdp.Env(**env_args, seed=seed + rank)

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "oh-et-al":
                raise NotImplementedError
            else:
                assert baseline is None
                recurrence = mdp.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainBandit(**kwargs).run()


def mdp_cli():
    parsers = build_parser()
    parser = parsers.main
    parsers.env.add_argument("--n-states", type=int, required=True)
    parsers.env.add_argument("--delayed-reward", action="store_true")
    parsers.agent.add_argument("--baseline", choices=["one-shot"])
    train_mdp(**hierarchical_parse_args(parser))
