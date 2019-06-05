# stdlib

from collections import ChainMap
# noinspection PyUnresolvedReferences
from pathlib import Path

import torch
from gym.wrappers import TimeLimit

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
import gridworld_env
from gridworld_env.subtasks_gridworld import SubtasksGridWorld  # noqa
from gridworld_env.subtasks_gridworld import get_task_space
from ppo.arguments import build_parser, get_args
from ppo.subtasks import SubtasksAgent, SubtasksTeacher
from ppo.train import Train
from ppo.wrappers import SubtasksWrapper, VecNormalize
from rl_utils import hierarchical_parse_args


def cli():
    Train(**get_args())


def class_parser(string):
    return dict(SubtasksGridWorld=SubtasksGridWorld)[string]


def make_subtasks_env(env_id, **kwargs):
    def helper(rank, seed, class_, max_episode_steps, **_kwargs):
        env = SubtasksWrapper(class_parser(class_)(**_kwargs))
        env.seed(seed + rank)
        print('Environment seed:', seed + rank)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=int(max_episode_steps))
        return env

    gridworld_args = gridworld_env.get_args(env_id)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return helper(**ChainMap(
        kwargs, gridworld_args
    ))  # combines kwargs and gridworld_args with preference for kwargs


def subtasks_cli():
    parser = build_parser()
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--task-types', nargs='*')
    task_parser.add_argument('--max-task-count', type=int)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int)
    kwargs = hierarchical_parse_args(parser)

    def train(task_args, **_kwargs):
        class TrainTeacher(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                return make_subtasks_env(
                    env_id=env_id, rank=rank, seed=seed, **task_args)

            # noinspection PyMethodOverriding
            @staticmethod
            def build_agent(envs, hidden_size, recurrent, entropy_coef, **_):
                return SubtasksAgent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    task_space=get_task_space(**task_args),
                    hidden_size=hidden_size,
                    entropy_coef=entropy_coef,
                    recurrent=recurrent,
                )

        TrainTeacher(**_kwargs)

    train(**kwargs)


def train_teacher_cli():
    parser = build_parser()
    parser.add_argument('--task-types', nargs='*')
    parser.add_argument('--max-task-count', type=int)
    parser.add_argument('--object-types', nargs='*')
    parser.add_argument('--n-subtasks', type=int)
    kwargs = hierarchical_parse_args(parser)

    def train(task_types, max_task_count, object_types, n_subtasks, **_kwargs):
        class TrainTeacher(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                return make_subtasks_env(
                    env_id=env_id,
                    task_types=task_types,
                    max_task_count=max_task_count,
                    object_types=object_types,
                    n_subtasks=n_subtasks,
                    rank=rank,
                    seed=seed)

            @staticmethod
            def build_agent(envs, **agent_args):
                return SubtasksTeacher(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    n_task_types=len(task_types),
                    n_objects=len(object_types),
                    **agent_args)

        TrainTeacher(**_kwargs)

    train(**kwargs)


def teach_cli():
    parser = build_parser()
    parser.add_argument('--behavior-agent-load-path', type=Path, required=True)
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--task-types', nargs='*')
    task_parser.add_argument('--max-task-count', type=int, required=True)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int, required=True)

    def train(env_id, task_args, ppo_args, behavior_agent_load_path, **kwargs):
        class TrainSubtasks(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                return make_subtasks_env(
                    env_id=env_id, rank=rank, seed=seed, **task_args)

            # noinspection PyMethodOverriding
            @staticmethod
            def build_agent(envs, hidden_size, recurrent, entropy_coef,
                            **agent_args):
                imitation_agent = SubtasksTeacher(
                    hidden_size=hidden_size,
                    recurrent=recurrent,
                    entropy_coef=entropy_coef,
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    n_task_types=len(task_args['task_types']),
                    n_objects=len(task_args['object_types']),
                    **agent_args)

                state_dict = torch.load(behavior_agent_load_path)
                imitation_agent.load_state_dict(state_dict['agent'])
                if isinstance(envs.venv, VecNormalize):
                    envs.venv.load_state_dict(state_dict['vec_normalize'])
                print(
                    f'Loaded behavior parameters from {behavior_agent_load_path}.'
                )

                return SubtasksAgent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    task_space=get_task_space(**task_args),
                    hidden_size=hidden_size,
                    entropy_coef=entropy_coef,
                    recurrent=recurrent,
                    imitation_agent=imitation_agent)

        # Train
        ppo_args.update(aux_loss_only=True)
        TrainSubtasks(env_id=env_id, ppo_args=ppo_args, **kwargs)

    train(**(hierarchical_parse_args(parser)))


if __name__ == "__main__":
    teach_cli()
