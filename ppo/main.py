# stdlib
import itertools
from pathlib import Path

# noinspection PyUnresolvedReferences
from gym.wrappers import TimeLimit
from rl_utils import hierarchical_parse_args

# noinspection PyUnresolvedReferences
import gridworld_env
from ppo.arguments import build_parser, get_args
from ppo.subtasks import SubtasksAgent
from ppo.train import Train
from ppo.wrappers import SubtasksWrapper


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
    parsers = build_parser()
    kwargs = hierarchical_parse_args(parsers)
    env_args = gridworld_env.get_args(kwargs['env_id'])

    def train(env_id, **_kwargs):
        class SubtasksTrainer(Train):
            @staticmethod
            def make_env(env_id, seed, rank, add_timestep):
                class_ = eval(_kwargs.pop('class'))
                env = SubtasksWrapper(class_(**_kwargs))
                env.seed(seed + rank)
                if 'max_episode_steps' in env_args:
                    env = TimeLimit(
                        env, max_episode_seconds=int(env_args['max_episode_steps']))
                return env

            # noinspection PyMethodOverriding
            @staticmethod
            def build_agent(envs, hidden_size, recurrent, **kwargs):
                return SubtasksAgent(envs.observation_space.shape,
                                     envs.action_space,
                                     (env_args['n_subtasks'], 3),
                                     hidden_size, recurrent)

        # Train
        SubtasksTrainer(env_id=env_id, **_kwargs)

    train(**kwargs)


if __name__ == "__main__":
    single_task_cli()
