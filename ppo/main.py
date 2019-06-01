# stdlib
import itertools
from pathlib import Path

# noinspection PyUnresolvedReferences
from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

import gridworld_env
# noinspection PyUnresolvedReferences
from gridworld_env import SubtasksGridWorld
from ppo.arguments import build_parser, get_args
from ppo.train import Trainer
from ppo.wrappers import SubtasksWrapper


def cli():
    Trainer(**get_args())


# def hsr_cli():
#     args = get_hsr_args()
#     env_wrapper(Trainer)(**args)


def teach_cli():
    parser = build_parser()
    parser.add_argument(
        '--save-dir',
        type=Path,
        help='directory to save agent logs.')
    task_parser = parser.add_argument_group('task_args')
    task_parser.add_argument('--task-types', nargs='*')
    task_parser.add_argument('--task-counts', nargs='*', type=int)
    task_parser.add_argument('--object-types', nargs='*')
    task_parser.add_argument('--n-subtasks', type=int)
    kwargs = hierarchical_parse_args(parser)
    env_args = gridworld_env.get_args(kwargs['env_id'])
    class_ = eval(env_args.pop('class'))
    max_episode_steps = env_args.pop('max_episode_steps', None)

    def task_iter(task_types, task_counts, object_types, n_subtasks):
        return itertools.product(task_types, task_counts, object_types,
                                 repeat=n_subtasks)

    def train(env_id, task_args, log_dir, load_path, save_dir, **_kwargs):
        for i, task in enumerate(task_iter(**task_args)):
            print('task', task)
            task = list(zip(*[iter(task)] * 3))
            log_dir = log_dir.joinpath(str(i))
            if not load_path and save_dir:
                load_path = save_dir.joinpath('checkpoint.pt')
                if not load_path.exists():
                    load_path = None

            class SubtasksTrainer(Trainer):
                @staticmethod
                def make_env(env_id, seed, rank, add_timestep):
                    env = SubtasksWrapper(class_(**env_args, task=task))
                    env.seed(seed + rank)
                    if max_episode_steps:
                        env = TimeLimit(env, max_episode_seconds=max_episode_steps)
                    return env

            SubtasksTrainer(
                log_dir=log_dir,
                load_path=load_path,
                save_dir=save_dir,
                env_id=env_id,
                **_kwargs)

    train(**kwargs)


if __name__ == "__main__":
    cli()
