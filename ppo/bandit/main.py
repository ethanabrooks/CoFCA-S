from rl_utils import hierarchical_parse_args

from ppo import bandit, gntm
from ppo.main import build_parser
from ppo.train import Train


def train_bandit(**kwargs):
    class TrainBandit(Train):
        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return bandit.bandit.Env(**env_args, seed=seed + rank)

        def build_agent(self, envs, entropy_coef=None, baseline=None, **agent_args):
            if baseline == "oh-et-al":
                recurrence = bandit.baselines.oh_et_al.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            else:
                assert baseline is None
                recurrence = bandit.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainBandit(**kwargs).run()


def bandit_cli():
    parsers = build_parser()
    parsers.env.add_argument("--n-lines", type=int, required=True)
    parsers.env.add_argument("--flip-prob", type=float, required=True)
    parsers.agent.add_argument("--baseline", choices=["oh-et-al"])
    train_bandit(**hierarchical_parse_args(parsers.main))
