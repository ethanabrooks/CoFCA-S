from rl_utils import hierarchical_parse_args

import ppo.agent
import ppo.graph_networks.agent
from ppo import graph_networks
from ppo.arguments import build_parser
from ppo.train import Train


def main(log_dir, baseline, seed, **kwargs):
    class _Train(Train):
        def build_agent(self, envs, debug=False, **agent_args):
            if baseline == "default":
                return ppo.agent.Agent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    **agent_args,
                )
            elif baseline == "oh-et-al":
                raise NotImplementedError
            return ppo.graph_networks.agent.Agent(
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                debug=debug,
                baseline=baseline,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return graph_networks.bandit.Env(
                **env_args, baseline=baseline, seed=seed + rank
            )

    _Train(**kwargs, seed=seed, log_dir=log_dir).run()


def bandit_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--baseline", choices=["oh-et-al", "default"])
    parsers.env.add_argument("--min-lines", type=int, required=True)
    parsers.env.add_argument("--max-lines", type=int, required=True)
    parsers.env.add_argument("--flip-prob", type=float, required=True)
    parsers.agent.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(bandit_args()))
