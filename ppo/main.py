from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import blocks_world, gntm
from ppo.train import Train
from gym.wrappers import TimeLimit


def build_parser():
    parsers = ppo.arguments.build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    return parsers


def train_blocks_world(
    time_limit, n_constraints, increment_curriculum_at_n_satisfied, **kwargs
):
    class TrainValues(Train):
        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return TimeLimit(
                blocks_world.Env(
                    **env_args, n_constraints=n_constraints, seed=seed + rank
                ),
                max_episode_steps=time_limit + n_constraints,
            )

        def run_epoch(self, *args, **kwargs):
            counter = super().run_epoch(*args, **kwargs)
            if (
                increment_curriculum_at_n_satisfied
                and counter["n_satisfied"] > increment_curriculum_at_n_satisfied
            ):
                self.envs.increment_curriculum()
            return counter

        def build_agent(
            self, envs, recurrent=None, entropy_coef=None, baseline=None, **agent_args
        ):
            if baseline == "one-shot":
                raise NotImplementedError
            else:
                assert baseline is None
                recurrence = blocks_world.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **agent_args,
                )
            return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)

    TrainValues(time_limit=time_limit + n_constraints, **kwargs).run()


def blocks_world_cli():
    parsers = build_parser()
    parsers.main.add_argument("--n-constraints", type=int, required=True)
    parsers.env.add_argument("--n-cols", type=int, required=True)
    parsers.main.add_argument("--increment-curriculum-at-n-satisfied", type=float)
    parsers.agent.add_argument("--num-slots", type=int, required=True)
    parsers.agent.add_argument("--slot-size", type=int, required=True)
    parsers.agent.add_argument("--embedding-size", type=int, required=True)
    parsers.agent.add_argument("--num-heads", type=int, required=True)
    train_blocks_world(**hierarchical_parse_args(parsers.main))


if __name__ == "__main__":
    blocks_world_cli()
