from pathlib import Path

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.agent
import ppo.control_flow.agent
import ppo.control_flow.env
import ppo.control_flow.multi_step.env
import ppo.control_flow.multi_step.minimal
import ppo.control_flow.multi_step.one_line
from ppo import control_flow
from ppo.arguments import build_parser
from ppo.train import Train

NAMES = ["instruction", "actions", "program_counter", "evaluations"]


def main(
    log_dir, seed, eval_lines, one_line, lower_level, lower_level_load_path, **kwargs
):
    if lower_level_load_path:
        lower_level = "pre-trained"

    class _Train(Train):
        def build_agent(self, envs, baseline=None, debug=False, **agent_args):
            obs_space = envs.observation_space
            ll_action_space = spaces.Discrete(
                ppo.control_flow.env.Action(*envs.action_space.nvec).lower
            )
            if lower_level == "train-alone":
                return ppo.agent.Agent(
                    lower_level=True,
                    obs_spaces=obs_space,
                    action_space=ll_action_space,
                    **agent_args,
                )
            agent_args.update(log_dir=log_dir)
            if baseline == "simple" or one_line:
                del agent_args["no_scan"]
                del agent_args["no_roll"]
                del agent_args["num_encoding_layers"]
                del agent_args["num_edges"]
                del agent_args["gate_coef"]
                del agent_args["no_op_coef"]
                return ppo.control_flow.multi_step.minimal.Agent(
                    observation_space=obs_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            del agent_args["recurrent"]
            return ppo.control_flow.agent.Agent(
                observation_space=obs_space,
                action_space=envs.action_space,
                eval_lines=eval_lines,
                debug=debug,
                baseline=baseline,
                lower_level=lower_level,
                lower_level_load_path=lower_level_load_path,
                **agent_args,
            )

        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, gridworld, **env_args
        ):
            args = dict(**env_args, eval_lines=eval_lines, seed=seed + rank, rank=rank)
            args["lower_level"] = lower_level
            del args["time_limit"]
            if one_line:
                return control_flow.multi_step.one_line.Env(**args)
            elif not gridworld:
                del args["max_while_objects"]
                del args["num_excluded_objects"]
                del args["temporal_extension"]
                return control_flow.env.Env(**args)
            else:
                return control_flow.multi_step.env.Env(**args)

        def process_infos(self, episode_counter, done, infos, **act_log):
            if lower_level != "train-alone":
                P = act_log.pop("P")
                P = P.transpose(0, 1)[done]
                if P.size(0) > 0:
                    P = P.cpu().numpy()
                    episode_counter["P"] += np.split(P, P.shape[0])
                for d in infos:
                    for name in NAMES:
                        if name in d:
                            episode_counter[name].append(d.pop(name))
            super().process_infos(episode_counter, done, infos, **act_log)

        def log_result(self, result: dict):
            if "subtasks_attempted" in result:
                subtasks_attempted = sum(result["subtasks_attempted"])
                if subtasks_attempted > 0:
                    result["success"] = (
                        sum(result["subtasks_complete"]) / subtasks_attempted
                    )
            if lower_level != "train-alone":
                names = NAMES + ["P"]
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

    _Train(**kwargs, seed=seed, log_dir=log_dir, time_limit=None).run()


def control_flow_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--eval-lines", type=int, nargs="*")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--one-line", action="store_true")
    parser.add_argument(
        "--lower-level", choices=["train-alone", "train-with-upper", "hardcoded"]
    )
    parser.add_argument("--lower-level-load-path")
    parsers.env.add_argument("--gridworld", action="store_true")
    ppo.control_flow.multi_step.env.build_parser(parsers.env)
    parsers.agent.add_argument("--lower-level-config", type=Path)
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--no-scan", action="store_true")
    parsers.agent.add_argument("--no-roll", action="store_true")
    parsers.agent.add_argument("--baseline")
    parsers.agent.add_argument("--hidden3", type=int, required=True)
    parsers.agent.add_argument("--hidden2", type=int, required=True)
    parsers.agent.add_argument("--conv-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--task-embed-size", type=int, required=True)
    parsers.agent.add_argument("--gate-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--gate-stride", type=int, required=True)
    parsers.agent.add_argument("--num-encoding-layers", type=int, required=True)
    parsers.agent.add_argument("--num-conv-layers", type=int, required=True)
    parsers.agent.add_argument("--num-edges", type=int, required=True)
    parsers.agent.add_argument("--gate-coef", type=float, required=True)
    parsers.agent.add_argument("--no-op-coef", type=float, required=True)
    parsers.agent.add_argument("--gate-pool-stride", type=int, required=True)
    parsers.agent.add_argument("--gate-pool-kernel-size", type=int, required=True)
    parsers.agent.add_argument("--gate-conv-kernel-size", type=int, required=True)
    parsers.agent.add_argument("--gate-conv-hidden-size", type=int, required=True)
    parsers.agent.add_argument("--concat", action="store_true")
    parsers.agent.add_argument("--kernel-size", type=int, required=True)
    parsers.agent.add_argument("--stride", type=int, required=True)
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(control_flow_args()))
