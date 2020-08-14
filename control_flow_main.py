from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import multi_step.env
import networks
import control_flow.agent
import control_flow
from env import Action
from main import add_arguments
from trainer import Trainer


def main(
    env_args,
    env_id,
    gridworld,
    log_dir,
    lower_level,
    lower_level_load_path,
    max_eval_lines,
    min_eval_lines,
    render,
    seed,
    **kwargs,
):
    if lower_level_load_path:
        lower_level = "pre-trained"

    class _Trainer(Trainer):
        def build_agent(self, envs, debug=False, **agent_args):
            obs_space = envs.observation_space
            ll_action_space = spaces.Discrete(Action(*envs.action_space.nvec).lower)
            if lower_level == "train-alone":
                return networks.Agent(
                    lower_level=True,
                    obs_spaces=obs_space,
                    action_space=ll_action_space,
                    **agent_args,
                )
            agent_args.update(log_dir=log_dir)
            del agent_args["recurrent"]
            del agent_args["num_conv_layers"]
            return control_flow.agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                eval_lines=max_eval_lines,
                debug=render and debug,
                lower_level=lower_level,
                lower_level_load_path=lower_level_load_path,
                **agent_args,
            )

        @staticmethod
        def make_env(seed, rank, evaluation, env_id=None):
            args = dict(
                **env_args,
                min_eval_lines=min_eval_lines,
                max_eval_lines=max_eval_lines,
                seed=seed + rank,
                rank=rank,
            )
            args["lower_level"] = lower_level
            args["break_on_fail"] = args["break_on_fail"] and render
            if not gridworld:
                del args["max_while_objects"]
                del args["num_excluded_objects"]
                del args["temporal_extension"]
                return multi_step.env.Env(**args)
            else:
                return multi_step.env.Env(**args)

        def process_infos(self, episode_counter, done, infos, **act_log):
            for d in infos:
                for k, v in d.items():
                    if k.startswith("cumulative_reward"):
                        episode_counter[k].append(v)
            super().process_infos(episode_counter, done, infos, **act_log)

        def _log_result(self, result: dict):
            keys = ["progress", "rewards", "instruction_len"]
            values = np.array(list(zip(*(iter(result[k]) for k in keys))))
            self.table.append(values)
            if "subtasks_attempted" in result:
                subtasks_attempted = sum(result["subtasks_attempted"])
                if subtasks_attempted > 0:
                    result["subtask_success"] = (
                        sum(result["subtasks_complete"]) / subtasks_attempted
                    )
            try:
                result["condition_evaluations"] = sum(
                    result["condition_evaluations"]
                ) / len(result["condition_evaluations"])
            except (KeyError, ZeroDivisionError):
                pass
            if lower_level != "train-alone":
                for name in names + ["eval_" + n for n in names]:
                    if name in result:
                        arrays = [x for x in result.pop(name) if x is not None]
                        if "P" not in name:
                            arrays = [np.array(x, dtype=int) for x in arrays]

                        np.savez(Path(self.log_dir, name), *arrays)

                for prefix in ("eval_", ""):
                    if prefix + "rewards" in result:
                        success = result[prefix + "rewards"]
                        np.save(Path(self.log_dir, prefix + "successes"), success)

            super().log_result(result)

    _Trainer.main(
        **kwargs, seed=seed, log_dir=log_dir, render=render, env_id="control-flow"
    )


def control_flow_args(parser):
    parsers = add_arguments(parser)
    parser = parsers.main
    parser.add_argument("--min-eval-lines", type=int, required=True)
    parser.add_argument("--max-eval-lines", type=int, required=True)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--lower-level", choices=["train-alone", "train-with-upper", "hardcoded"]
    )
    parser.add_argument("--lower-level-load-path")
    parser.add_argument("--gridworld", action="store_true")
    multi_step.env.build_parser(parser.add_argument_group("env_args"))
    parsers.agent.add_argument("--lower-level-config", type=Path)
    parsers.agent.add_argument("--no-debug", dest="debug", action="store_false")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--no-pointer", action="store_true")
    parsers.agent.add_argument("--olsk", action="store_true")
    parsers.agent.add_argument("--transformer", action="store_true")
    parsers.agent.add_argument("--fuzz", action="store_true")
    parsers.agent.add_argument("--sum-pool", action="store_true")
    parsers.agent.add_argument("--conv-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--task-embed-size", type=int, required=True)
    parsers.agent.add_argument("--lower-embed-size", type=int, required=True)
    parsers.agent.add_argument("--inventory-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--num-conv-layers", type=int, required=True)
    parsers.agent.add_argument("--num-edges", type=int, required=True)
    parsers.agent.add_argument("--gate-coef", type=float, required=True)
    parsers.agent.add_argument("--no-op-coef", type=float, required=True)
    parsers.agent.add_argument("--kernel-size", type=int, required=True)
    parsers.agent.add_argument("--stride", type=int, required=True)
    return parser


if __name__ == "__main__":
    PARSER = ArgumentParser()
    main(**hierarchical_parse_args(control_flow_args(PARSER)))
