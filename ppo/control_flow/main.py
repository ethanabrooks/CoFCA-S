import functools

from rl_utils import hierarchical_parse_args
import numpy as np

import ppo.agent
import ppo.control_flow.agent
import ppo.control_flow.env
import ppo.control_flow.multi_step.env
from ppo import control_flow
from ppo.arguments import build_parser
from ppo.control_flow.multi_step.env import Env
from ppo.train import Train
import torch


def main(log_dir, seed, max_lines, eval_lines, **kwargs):
    class _Train(Train):
        def __init__(self, min_lines, load_path, env_args, **kwargs):
            if load_path:
                self.n_lines = torch.load(load_path)["n_lines"]
            else:
                self.n_lines = min_lines
            env_args.update(min_lines=min_lines)
            super().__init__(**kwargs, env_args=env_args, load_path=load_path)

        def build_agent(self, envs, debug=False, **agent_args):
            obs_space = envs.observation_space
            return ppo.control_flow.agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                max_train_lines=max_lines,
                eval_lines=eval_lines,
                debug=debug,
                **agent_args,
            )

        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, world_size, **env_args
        ):
            args = dict(
                **env_args,
                eval_lines=eval_lines,
                max_lines=max_lines,
                baseline=False,
                seed=seed + rank,
            )
            if world_size is None:
                return control_flow.env.Env(**args)
            else:
                return Env(**args, world_size=world_size)

        def make_vec_envs(self, use_monkey, **kwargs):
            # noinspection PyAttributeOutsideInit
            if use_monkey:
                if "monkey" not in Env.line_objects:
                    Env.line_objects.append("monkey")
            elif "greenbot" in Env.subtask_objects:
                Env.subtask_objects.remove("greenbot")
                Env.world_objects.remove("greenbot")

            # noinspection PyAttributeOutsideInit
            return super().make_vec_envs(**kwargs)

        def build_envs_thunk(self, min_lines, **kwargs):
            return functools.partial(self.make_vec_envs, **kwargs)

        def run_epoch(self, **kwargs):
            result = super().run_epoch(**kwargs)
            if "rewards" in result and result["rewards"]:
                result.update(
                    cumulative_reward=self.n_lines - 1 + np.mean(result["rewards"])
                )
            return result

        def increment_envs(self):
            self.n_lines = min(self.n_lines + 1, max_lines)
            return self.envs_thunk(min_lines=self.n_lines)

        def get_save_dict(self):
            return dict(n_lines=self.n_lines, **super().get_save_dict())

    _Train(**kwargs, seed=seed, log_dir=log_dir).run()


def bandit_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--eval-lines", type=int, required=True)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--min-lines", type=int, required=True)
    parser.add_argument("--max-lines", type=int, required=True)
    ppo.control_flow.env.build_parser(parsers.env)
    parsers.env.add_argument("--world-size", type=int)
    parsers.env.add_argument("--use-monkey", action="store_true")
    parsers.env.add_argument("--soft-increment", action="store_true")
    parsers.env.add_argument("--add-while-obj-prob", type=float, required=True)
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--no-pointer", action="store_true")
    parsers.agent.add_argument("--include-action", action="store_true")
    parsers.agent.add_argument("--conv-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--encoder-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--num-encoding-layers", type=int, required=True)
    parsers.agent.add_argument("--kernel-size", type=int, required=True)
    parsers.agent.add_argument("--num-edges", type=int, required=True)
    parsers.agent.add_argument("--gate-coef", type=float)
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(bandit_args()))
