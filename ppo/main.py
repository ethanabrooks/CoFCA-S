# stdlib
# this is as test

# noinspection PyUnresolvedReferences
from collections import ChainMap
from pathlib import Path

from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import gntm
from ppo.blocks_world import dnc, planner
from ppo.train import Train
from ppo.wrappers import VecNormalize


def add_task_args(parser):
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--interactions', nargs='*')
    task_parser.add_argument('--max-task-count', type=int, required=True)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int, required=True)


def add_env_args(parser):
    env_parser = parser.add_argument_group('env_args')
    env_parser.add_argument('--min-objects', type=int, required=True)
    env_parser.add_argument('--debug', action='store_true')
    env_parser.add_argument(
        '--eval-subtask', dest='eval_subtasks', default=[], type=int, nargs=3, action='append')


def cli():
    Train(**get_args())


def get_spaces(envs, control_flow):
    obs_spaces = envs.observation_space.spaces
    if control_flow:
        obs_spaces = ppo.control_flow.Obs(*obs_spaces)
    else:
        obs_spaces = ppo.subtasks.Obs(*obs_spaces)
    return obs_spaces


def make_subtasks_env(env_id, **kwargs):
    def helper(seed, rank, control_flow, max_episode_steps, class_, debug, **_kwargs):
        if rank == 1:
            print('Environment args:')
            for k, v in _kwargs.items():
                print(f'{k:20}{v}')
        if control_flow:
            env = ppo.control_flow.Wrapper(ControlFlowGridWorld(**_kwargs))
        else:
            env = ppo.subtasks.Wrapper(SubtasksGridWorld(**_kwargs))
        if debug:
            env = ppo.subtasks.DebugWrapper(env)
        env.seed(seed + rank)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=int(max_episode_steps))
        return env

    gridworld_args = gridworld_env.get_args(env_id)
    kwargs.update(add_timestep=None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return helper(**ChainMap(
        kwargs, gridworld_args))  # combines kwargs and gridworld_args with preference for kwargs


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


            @staticmethod
            def build_agent(envs, **agent_args):
                obs_spaces = get_spaces(envs, control_flow)
                agent_args = dict(
                    obs_spaces=obs_spaces, action_space=envs.action_space, **agent_args)
                if student:
                    return ppo.subtasks.Student(**agent_args, **student_args)
                else:
                    return ppo.subtasks.Teacher(**agent_args)

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
    subtasks_parser.add_argument('--subtasks-hidden-size', type=int, required=True)
    subtasks_parser.add_argument('--subtasks-entropy-coef', type=float, required=True)
    subtasks_parser.add_argument('--subtasks-recurrent', action='store_true')
    subtasks_parser.add_argument('--hard-update', action='store_true')
    subtasks_parser.add_argument('--multiplicative-interaction', action='store_true')

    def train(env_id, task_args, ppo_args, agent_load_path, subtasks_args, env_args, control_flow,
              **kwargs):
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
                    agent = ppo.subtasks.Teacher(
                        obs_spaces=obs_spaces, action_space=envs.action_space, **agent_args)

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

                _subtasks_args = {k.replace('subtasks_', ''): v for k, v in subtasks_args.items()}

                metacontroller_kwargs = dict(
                    obs_spaces=obs_spaces,
                    action_space=envs.action_space,
                    planning_steps=planning_steps,
                    **planner_args,
                    **dnc_args,
                    **agent_args,
                )
                return planner.Agent(
                    entropy_coef=entropy_coef,
                    model_loss_coef=model_loss_coef,
                    recurrence=recurrence,
                )

    TrainValues(**kwargs).run()


def blocks_world_cli():
    parsers = build_parser()
    parsers.main.add_argument("--baseline", choices=["dnc"])
    parsers.main.add_argument("--planning-steps", type=int, default=10)
    parsers.main.add_argument("--increment-curriculum-at-n-satisfied", type=float)
    parsers.env.add_argument("--n-cols", type=int, required=True)
    parsers.main.add_argument("--increment-curriculum-at-n-satisfied", type=float)
    parsers.agent.add_argument("--num-slots", type=int, required=True)
    parsers.agent.add_argument("--slot-size", type=int, required=True)
    parsers.agent.add_argument("--embedding-size", type=int, required=True)
    parsers.agent.add_argument("--model-loss-coef", type=float, required=True)
    planner_parser = parsers.agent.add_argument_group("planner_args")
    planner_parser.add_argument("--num-model-layers", type=int)
    planner_parser.add_argument("--num-embedding-layers", type=int)
    dnc_parser = parsers.agent.add_argument_group("dnc_args")
    dnc_parser.add_argument("--num-slots", type=int)
    dnc_parser.add_argument("--slot-size", type=int)
    dnc_parser.add_argument("--num-heads", type=int)
    train_blocks_world(**hierarchical_parse_args(parsers.main))


if __name__ == "__main__":
    blocks_world_cli()
