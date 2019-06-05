# stdlib

# noinspection PyUnresolvedReferences
from gym.wrappers import TimeLimit

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
import gridworld_env
from gridworld_env.subtasks_gridworld import SubtasksGridWorld  # noqa
from gridworld_env.subtasks_gridworld import get_task_space
from ppo.arguments import build_parser, get_args
from ppo.subtasks import SubtasksAgent, SubtasksTeacher
from ppo.train import Train
from ppo.wrappers import SubtasksWrapper
from rl_utils import hierarchical_parse_args


def cli():
    Train(**get_args())


def class_parser(str):
    return dict(SubtasksGridWorld=SubtasksGridWorld)[str]


def _make_subtasks_env(rank, seed, class_, max_episode_steps, **kwargs):
    env = SubtasksWrapper(class_parser(class_)(**kwargs))
    env.seed(seed + rank)
    print('Environment seed:', seed + rank)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_seconds=int(max_episode_steps))
    return env


def make_subtasks_env(env_id, **kwargs):
    gridworld_args = gridworld_env.get_args(env_id)
    gridworld_args.update(**kwargs)
    return _make_subtasks_env(**gridworld_args)


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
            def build_behavior_agent(envs, **agent_args):
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
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--task-types', nargs='*')
    task_parser.add_argument('--max-task-count', type=int)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int)

    def train(env_id, task_args, ppo_args, **kwargs):
        gridworld_args = gridworld_env.get_args(env_id)
        max_episode_steps = gridworld_args.pop('max_episode_steps', None)
        task_args = {k: v for k, v in task_args.items() if v}
        gridworld_args.update(**task_args)
        class_ = eval(gridworld_args.pop('class'))

        class TrainSubtasks(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                return make_subtasks_env(
                    rank=rank,
                    seed=seed,
                    class_=class_,
                    max_episode_steps=max_episode_steps,
                    **gridworld_args)

            # noinspection PyMethodOverriding
            @staticmethod
            def build_agent(envs, hidden_size, recurrent, entropy_coef,
                            **_):
                return SubtasksAgent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    task_space=get_task_space(**task_args),
                    hidden_size=hidden_size,
                    entropy_coef=entropy_coef,
                    recurrent=recurrent)

            @staticmethod
            def build_behavior_agent(envs, **agent_args):
                return SubtasksTeacher(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    n_task_types=len(task_args['task_types']),
                    n_objects=len(task_args['object_types']),
                    **agent_args)

        # Train
        ppo_args.update(aux_loss_only=True)
        TrainSubtasks(env_id=env_id, ppo_args=ppo_args, **kwargs)

    train(**(hierarchical_parse_args(parser)))


if __name__ == "__main__":
    teach_cli()
