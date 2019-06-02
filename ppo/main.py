# stdlib
import itertools
from pathlib import Path

# noinspection PyUnresolvedReferences
from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

# noinspection PyUnresolvedReferences
import gridworld_env
from ppo.arguments import build_parser, get_args
from ppo.train import Trainer
from ppo.wrappers import SubtasksWrapper


def cli():
    Trainer(**get_args())


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
        class SubtasksTrainer(Trainer):
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
    kwargs = hierarchical_parse_args(parser)
    env_args = gridworld_env.get_args(kwargs['env_id'])
    class_ = eval(env_args.pop('class'))
    max_episode_steps = env_args.pop('max_episode_steps', None)

    def train(env_id, **_kwargs):

        class SubtasksTrainer(Trainer):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                env = SubtasksWrapper(class_(**env_args, task=task))
                env.seed(seed + rank)
                if max_episode_steps:
                    env = TimeLimit(
                        env, max_episode_seconds=max_episode_steps)
                return env

        SubtasksTrainer(
            env_id=env_id,
            **_kwargs)

    train(**kwargs)


if __name__ == "__main__":
    single_task_cli()
