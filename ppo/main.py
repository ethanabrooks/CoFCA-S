from collections import ChainMap
from pathlib import Path

from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

import gridworld_env
from gridworld_env.control_flow_gridworld import ControlFlowGridworld, TaskTypes
import gridworld_env.matrix_control_flow_gridworld
import gridworld_env.subtasks_gridworld
import ppo
from ppo.arguments import build_parser, get_args
import ppo.control_flow.agent
import ppo.control_flow.analogy_learner
import ppo.control_flow.lower_level
import ppo.matrix_control_flow
from ppo.train import Train


def add_task_args(parser):
    task_parser = parser.add_argument_group("task_args")
    task_parser.add_argument("--interactions", nargs="*")
    task_parser.add_argument("--max-task-count", type=int, required=True)
    task_parser.add_argument("--object-types", nargs="*")
    task_parser.add_argument("--n-subtasks", type=int, required=True)


def add_env_args(parser):
    env_parser = parser.add_argument_group("env_args")
    env_parser.add_argument("--min-objects", type=int, required=True)
    env_parser.add_argument("--task-type", type=lambda s: TaskTypes[s], default="Auto")
    env_parser.add_argument("--max-loops", type=int)
    env_parser.add_argument(
        "--eval-subtask",
        dest="eval_subtasks",
        default=[],
        type=int,
        nargs=3,
        action="append",
    )


def cli():
    Train(**get_args())


def make_subtasks_env(env_id, **kwargs):
    def helper(seed, rank, max_episode_steps, class_, **_kwargs):
        if rank == 1:
            print("Environment args:")
            for k, v in _kwargs.items():
                print(f"{k:20}{v}")
        env = ppo.control_flow.Wrapper(
            TimeLimit(
                ControlFlowGridworld(**_kwargs),
                max_episode_steps=int(max_episode_steps),
            )
        )
        # if debug:
        #     env = ppo.control_flow.DebugWrapper(env)
        env.seed(seed + rank)
        return env

    gridworld_args = gridworld_env.get_args(env_id)
    kwargs.update(add_timestep=None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    chain_map = ChainMap(kwargs, gridworld_args)

    return helper(
        **chain_map
    )  # combines kwargs and gridworld_args with preference for kwargs


def train_lower_level_cli(student):
    parser = build_parser()
    add_task_args(parser)
    add_env_args(parser)
    if student:
        student_parser = parser.add_argument_group("student_args")
        student_parser.add_argument("--embedding-dim", type=int, required=True)
        student_parser.add_argument("--tau-diss", type=float, required=True)
        student_parser.add_argument("--tau-diff", type=float, required=True)
        student_parser.add_argument("--xi", type=float, required=True)
    kwargs = hierarchical_parse_args(parser)

    def train(task_args, env_args, student_args=None, **_kwargs):
        class TrainSkill(Train):
            @staticmethod
            def make_env(add_timestep, **make_env_args):
                return make_subtasks_env(
                    **env_args,
                    **make_env_args,
                    **task_args,
                    max_episode_steps=kwargs["max_episode_steps"],
                )

            @staticmethod
            def build_agent(envs, **agent_args):
                agent_args = dict(
                    obs_spaces=envs.obs_spaces,
                    action_space=envs.action_space,
                    **agent_args,
                )
                if student:
                    return ppo.control_flow.AnalogyLearner(**agent_args, **student_args)
                else:
                    return ppo.control_flow.LowerLevel(**agent_args)

        TrainSkill(**_kwargs)

    train(**kwargs)


def teacher_cli():
    train_lower_level_cli(student=False)


def student_cli():
    train_lower_level_cli(student=True)


def metacontroller_cli():
    parser = build_parser()
    add_task_args(parser)
    add_env_args(parser)
    subtasks_parser = parser.add_argument_group("subtasks_args")
    subtasks_parser.add_argument("--agent-load-path", type=Path)
    subtasks_parser.add_argument(
        "--metacontroller-hidden-size", type=int, required=True
    )
    subtasks_parser.add_argument("--g-entropy-coef", type=float, required=True)
    subtasks_parser.add_argument("--z-entropy-coef", type=float, required=True)
    subtasks_parser.add_argument("--l-entropy-coef", type=float, required=True)
    subtasks_parser.add_argument("--cr-entropy-coef", type=float, required=True)
    subtasks_parser.add_argument("--cg-entropy-coef", type=float, required=True)
    subtasks_parser.add_argument("--metacontroller-recurrent", action="store_true")
    subtasks_parser.add_argument("--debug", action="store_true")

    def train(env_id, task_args, ppo_args, subtasks_args, env_args, **kwargs):
        class TrainSubtasks(Train):
            @staticmethod
            def make_env(**_kwargs):
                return make_subtasks_env(
                    **env_args,
                    **_kwargs,
                    **task_args,
                    max_episode_steps=kwargs["max_episode_steps"],
                )

            # noinspection PyMethodOverriding
            def build_agent(self, envs, **agent_args):
                metacontroller_kwargs = dict(
                    obs_space=envs.observation_space,
                    action_space=envs.action_space,
                    agent_args=agent_args,
                    **{
                        k.replace("metacontroller_", ""): v
                        for k, v in subtasks_args.items()
                    },
                )
                return ppo.control_flow.Agent(**metacontroller_kwargs)

        # ppo_args.update(aux_loss_only=True)
        TrainSubtasks(env_id=env_id, ppo_args=ppo_args, **kwargs)

    train(**(hierarchical_parse_args(parser)))


if __name__ == "__main__":
    metacontroller_cli()
