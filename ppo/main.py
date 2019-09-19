import numpy as np
from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import gntm
from ppo.blocks_world import dnc, planner
from ppo.train import Train


def build_parser():
    parsers = ppo.arguments.build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    return parsers


def train_blocks_world(increment_curriculum_at_n_satisfied, **kwargs):
    class TrainValues(Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            return blocks_world.Env(**env_args, seed=seed + rank)

        def run_epoch(self, *args, **kwargs):
            counter = super().run_epoch(*args, **kwargs)
            if (
                increment_curriculum_at_n_satisfied
                and counter["n_satisfied"] > increment_curriculum_at_n_satisfied
            ):
                self.envs.increment_curriculum()
            return counter

        def build_agent(
            self,
            envs,
            recurrent=None,
            entropy_coef=None,
            model_loss_coef=None,
            dnc_args=None,
            planner_args=None,
            **agent_args,
        ):
            if baseline == "dnc":
                recurrence = dnc.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    **dnc_args,
                    **agent_args,
                )
                return gntm.Agent(entropy_coef=entropy_coef, recurrence=recurrence)
            else:
                assert baseline is None
                recurrence = planner.Recurrence(
                    observation_space=envs.observation_space,
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
