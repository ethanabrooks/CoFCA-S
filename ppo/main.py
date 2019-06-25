# stdlib

# noinspection PyUnresolvedReferences
from collections import ChainMap
from pathlib import Path

from gym.wrappers import TimeLimit
import torch

import gridworld_env
from gridworld_env.control_flow_gridworld import ControlFlowGridWorld
from gridworld_env.subtasks_gridworld import SubtasksGridWorld
import ppo
from ppo.arguments import build_parser, get_args
import ppo.control_flow.agent
import ppo.control_flow.wrappers
import ppo.subtasks.agent
import ppo.subtasks.student
import ppo.subtasks.teacher
from ppo.subtasks.wrappers import Wrapper
from ppo.train import Train
from ppo.wrappers import VecNormalize
from rl_utils import hierarchical_parse_args


def add_task_args(parser):
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--interactions', nargs='*')
    task_parser.add_argument('--max-task-count', type=int, required=True)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int, required=True)


def add_env_args(parser):
    env_parser = parser.add_argument_group('env_args')
    env_parser.add_argument('--n-objects', type=int, required=True)
    env_parser.add_argument(
        '--eval-subtask',
        dest='eval_subtasks',
        default=[],
        type=int,
        nargs=3,
        action='append')


def cli():
    Train(**get_args())


def get_spaces(envs, control_flow):
    obs_spaces = envs.observation_space.spaces
    if control_flow:
        obs_spaces = ppo.control_flow.wrappers.Obs(*obs_spaces)
    else:
        obs_spaces = ppo.subtasks.wrappers.Obs(*obs_spaces)
    return obs_spaces


def make_subtasks_env(env_id, **kwargs):
    def helper(seed, rank, control_flow, max_episode_steps, class_, **_kwargs):
        if rank == 1:
            print('Environment args:')
            for k, v in _kwargs.items():
                print(f'{k:20}{v}')
        if control_flow:
            env = ppo.control_flow.wrappers.Wrapper(
                ControlFlowGridWorld(**_kwargs))
        else:
            env = ppo.subtasks.wrappers.Wrapper(SubtasksGridWorld(**_kwargs))
        env.seed(seed + rank)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=int(max_episode_steps))
        return env

    gridworld_args = gridworld_env.get_args(env_id)
    kwargs.update(add_timestep=None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return helper(**ChainMap(
        kwargs, gridworld_args
    ))  # combines kwargs and gridworld_args with preference for kwargs


def train_lower_level_cli(student):
    parser = build_parser()
    parser.add_argument('--control-flow', action='store_true')
    add_task_args(parser)
    add_env_args(parser)
    if student:
        student_parser = parser.add_argument_group('student_args')
        student_parser.add_argument('--embedding-dim', type=int, required=True)
        student_parser.add_argument('--tau-diss', type=float, required=True)
        student_parser.add_argument('--tau-diff', type=float, required=True)
        student_parser.add_argument('--xi', type=float, required=True)
    kwargs = hierarchical_parse_args(parser)

    def train(task_args, env_args, control_flow, student_args=None, **_kwargs):
        class TrainSkill(Train):
            @staticmethod
            def make_env(add_timestep, **make_env_args):
                return make_subtasks_env(
                    **env_args,
                    **make_env_args,
                    **task_args,
                    max_episode_steps=kwargs['max_episode_steps'],
                    control_flow=control_flow)

            @staticmethod
            def build_agent(envs, **agent_args):
                obs_spaces = get_spaces(envs, control_flow)
                agent_args = dict(
                    obs_spaces=obs_spaces,
                    action_space=envs.action_space,
                    **agent_args)
                if student:
                    return ppo.subtasks.student.Student(
                        **agent_args, **student_args)
                else:
                    return ppo.subtasks.teacher.Teacher(**agent_args)

        TrainSkill(**_kwargs)

    train(**kwargs)


def teacher_cli():
    train_lower_level_cli(student=False)


def student_cli():
    train_lower_level_cli(student=True)


def metacontroller_cli():
    parser = build_parser()
    parser.add_argument('--agent-load-path', type=Path)
    parser.add_argument('--control-flow', action='store_true')
    add_task_args(parser)
    add_env_args(parser)
    subtasks_parser = parser.add_argument_group('subtasks_args')
    subtasks_parser.add_argument(
        '--subtasks-hidden-size', type=int, required=True)
    subtasks_parser.add_argument(
        '--subtasks-entropy-coef', type=float, required=True)
    subtasks_parser.add_argument('--subtasks-recurrent', action='store_true')
    subtasks_parser.add_argument('--hard-update', action='store_true')
    subtasks_parser.add_argument(
        '--multiplicative-interaction', action='store_true')

    def train(env_id, task_args, ppo_args, agent_load_path, subtasks_args,
              env_args, control_flow, **kwargs):
        class TrainSubtasks(Train):
            @staticmethod
            def make_env(**_kwargs):
                return make_subtasks_env(
                    **env_args,
                    **_kwargs,
                    **task_args,
                    max_episode_steps=kwargs['max_episode_steps'],
                    control_flow=control_flow)

            # noinspection PyMethodOverriding
            @staticmethod
            def build_agent(envs, **agent_args):
                agent = None
                obs_spaces = get_spaces(envs, control_flow)
                if agent_load_path:
                    agent = ppo.subtasks.teacher.Teacher(
                        obs_spaces=obs_spaces,
                        action_space=envs.action_space,
                        **agent_args)

                    state_dict = torch.load(agent_load_path)
                    state_dict['agent'].update(
                        part0_one_hot=agent.part0_one_hot,
                        part1_one_hot=agent.part1_one_hot,
                        part2_one_hot=agent.part2_one_hot,
                    )
                    agent.load_state_dict(state_dict['agent'])
                    if isinstance(envs.venv, VecNormalize):
                        # noinspection PyUnresolvedReferences
                        envs.venv.load_state_dict(state_dict['vec_normalize'])
                    print(f'Loaded teacher parameters from {agent_load_path}.')

                _subtasks_args = {
                    k.replace('subtasks_', ''): v
                    for k, v in subtasks_args.items()
                }

                metacontroller_kwargs = dict(
                    obs_spaces=obs_spaces,
                    action_space=envs.action_space,
                    agent=agent,
                    **_subtasks_args)
                if control_flow:
                    return ppo.control_flow.agent.Agent(
                        **metacontroller_kwargs)
                else:
                    return ppo.subtasks.agent.Agent(**metacontroller_kwargs)

        # ppo_args.update(aux_loss_only=True)
        TrainSubtasks(env_id=env_id, ppo_args=ppo_args, **kwargs)

    train(**(hierarchical_parse_args(parser)))


if __name__ == "__main__":
    metacontroller_cli()
