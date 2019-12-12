from gym.spaces import Box
from rl_utils import hierarchical_parse_args

import ppo.agent
import ppo.control_flow.agent
import ppo.control_flow.env
import ppo.control_flow.multi_step.env
from ppo import control_flow
from ppo.arguments import build_parser
from ppo.train import Train


def main(log_dir, seed, eval_lines, **kwargs):
    class _Train(Train):
        def build_agent(self, envs, debug=False, **agent_args):
            obs_space = envs.observation_space
            return ppo.control_flow.agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                eval_lines=eval_lines,
                debug=debug,
                **agent_args,
            )

        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, world_size, **env_args
        ):
            args = dict(
                **env_args, eval_lines=eval_lines, baseline=False, seed=seed + rank
            )
            if world_size is None:
                return control_flow.env.Env(**args)
            else:
                return control_flow.multi_step.env.Env(**args, world_size=world_size)

    _Train(**kwargs, seed=seed, log_dir=log_dir).run()


def bandit_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--eval-lines", type=int, required=True)
    parser.add_argument("--no-eval", action="store_true")
    ppo.control_flow.env.build_parser(parsers.env)
    parsers.env.add_argument("--world-size", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--no-pointer", action="store_true")
    parsers.agent.add_argument("--include-action", action="store_true")
    parsers.agent.add_argument("--num-encoding-layers", type=int, required=True)
    parsers.agent.add_argument("--num-edges", type=int, required=True)
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(bandit_args()))
