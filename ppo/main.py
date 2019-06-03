# stdlib

# noinspection PyUnresolvedReferences
from gym.wrappers import TimeLimit

# noinspection PyUnresolvedReferences
import gridworld_env
# noinspection PyUnresolvedReferences
from gridworld_env.subtasks_gridworld import get_task_space, SubtasksGridWorld  # noqa
from ppo.arguments import build_parser, get_args
from ppo.subtasks import SubtasksAgent
from ppo.train import Train
from ppo.wrappers import SubtasksWrapper
from rl_utils import hierarchical_parse_args


def cli():
    Train(**get_args())


# def hsr_cli():
#     args = get_hsr_args()
#     env_wrapper(Trainer)(**args)


def single_task_cli():
    parser = build_parser()
    parser.add_argument('--subtask', dest='task', action='append')
    kwargs = hierarchical_parse_args(parser)
    env_args = gridworld_env.get_args(kwargs['env_id'])
    class_ = eval(env_args.pop('class'))
    max_episode_steps = env_args.pop('max_episode_steps', None)

    def train(env_id, task, **_kwargs):
        class SubtasksTrainer(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                env = SubtasksWrapper(class_(**env_args, task=task))
                env.seed(seed + rank)
                if max_episode_steps:
                    env = TimeLimit(env, max_episode_seconds=max_episode_steps)
                return env

        SubtasksTrainer(env_id=env_id, **_kwargs)

    train(**kwargs)


def teach_cli():
    parser = build_parser()
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--task-types', nargs='*')
    task_parser.add_argument('--max-task-count', type=int)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int)
    kwargs = hierarchical_parse_args(parser)
    gridworld_args = gridworld_env.get_args(kwargs['env_id'])
    class_ = eval(gridworld_args.pop('class'))

    def train(env_id, task_args, **_kwargs):
        max_episode_steps = gridworld_args.pop('max_episode_steps', None)
        task_args = {
            k: v if v else gridworld_args[k]
            for k, v in task_args.items()
        }
        gridworld_args.update(**task_args)

        class TrainSubtasks(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                env = SubtasksWrapper(class_(**gridworld_args))
                env.seed(seed + rank)
                if max_episode_steps is not None:
                    env = TimeLimit(
                        env, max_episode_seconds=int(max_episode_steps))
                return env

            # noinspection PyMethodOverriding
            @staticmethod
            def build_agent(envs, hidden_size, recurrent, entropy_coef, **kwargs):
                return SubtasksAgent(
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    task_space=get_task_space(**task_args),
                    hidden_size=hidden_size,
                    entropy_coef=entropy_coef,
                    recurrent=recurrent)

        # Train
        TrainSubtasks(env_id=env_id, **_kwargs)

    train(**kwargs)


if __name__ == "__main__":
    teach_cli()
