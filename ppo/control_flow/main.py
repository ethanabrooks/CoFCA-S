from rl_utils import hierarchical_parse_args

import ppo.agent
import ppo.control_flow.agent
import ppo.control_flow.env
from ppo import control_flow
from ppo.arguments import build_parser
from ppo.train import Train


def main(log_dir, seed, **kwargs):
    class _Train(Train):
        def build_agent(self, envs, debug=False, **agent_args):
            return ppo.control_flow.agent.Agent(
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                debug=debug,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return control_flow.env.Env(**env_args, seed=seed + rank)

    _Train(**kwargs, seed=seed, log_dir=log_dir).run()


def bandit_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--no-eval", action="store_true")
    parsers.env.add_argument("--min-lines", type=int, required=True)
    parsers.env.add_argument("--max-lines", type=int, required=True)
    parsers.env.add_argument("--num-subtasks", type=int, default=12)
    parsers.env.add_argument("--no-op-limit", type=int)
    parsers.env.add_argument("--eval-lines", type=int)
    parsers.env.add_argument("--flip-prob", type=float, required=True)
    parsers.env.add_argument("--delayed-reward", action="store_true")
    parsers.env.add_argument("--eval-condition-size", action="store_true")
    parsers.env.add_argument("--max-nesting-depth", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--num-encoding-layers", type=int, required=True)
    parsers.agent.add_argument("--num-edges", type=int, required=True)
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(bandit_args()))
