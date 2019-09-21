import numpy as np
from rl_utils import hierarchical_parse_args

import ppo.arguments
import ppo.bandit.baselines.oh_et_al
import ppo.maze.baselines
from ppo import gntm
from ppo.agent import Agent
from ppo.blocks_world import dnc, single_step, non_recurrent
from ppo.train import Train


def build_parser():
    parsers = ppo.arguments.build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parsers.agent.add_argument("--debug", action="store_true")
    return parsers


def train_blocks_world(
    increment_curriculum_at_n_satisfied, planning_steps, baseline, **_kwargs
):
    class TrainValues(Train):
        @staticmethod
        def make_env(
            seed, rank, evaluation, env_id, add_timestep, time_limit, **env_args
        ):
            if baseline == "dnc":
                return dnc.Env(**env_args, seed=seed + rank)
            elif baseline == "non-recurrent":
                return non_recurrent.Env(
                    **env_args, planning_steps=planning_steps, seed=seed + rank
                )
            else:
                assert baseline is None
                # return planner.Env(
                return single_step.Env(
                    **env_args, planning_steps=planning_steps, seed=seed + rank
                )

        def run_epoch(self, *args, **kwargs):
            dictionary = super().run_epoch(*args, **kwargs)
            try:
                increment_curriculum = (
                    np.mean(dictionary["n_satisfied"])
                    > increment_curriculum_at_n_satisfied
                )
            except (TypeError, KeyError):
                increment_curriculum = False
            if increment_curriculum:
                self.envs.increment_curriculum()
            return dictionary

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
            elif baseline == "non-recurrent":
                del agent_args["debug"]
                del agent_args["embedding_size"]
                return Agent(
                    entropy_coef=entropy_coef,
                    obs_shape=envs.observation_space.shape,
                    action_space=envs.action_space,
                    recurrent=False,
                    **agent_args,
                )
            else:
                assert baseline is None
                # recurrence = planner.Recurrence(
                recurrence = single_step.Recurrence(
                    observation_space=envs.observation_space,
                    action_space=envs.action_space,
                    planning_steps=planning_steps,
                    **planner_args,
                    **dnc_args,
                    **agent_args,
                )
                # return planner.Agent(
                return single_step.Agent(
                    entropy_coef=entropy_coef,
                    model_loss_coef=model_loss_coef,
                    recurrence=recurrence,
                )

    TrainValues(**_kwargs).run()


def blocks_world_cli():
    parsers = build_parser()
    parsers.main.add_argument("--baseline", choices=["dnc", "non-recurrent"])
    parsers.main.add_argument("--planning-steps", type=int, default=10)
    parsers.main.add_argument("--increment-curriculum-at-n-satisfied", type=float)
    parsers.env.add_argument("--n-cols", type=int, required=True)
    parsers.env.add_argument("--curriculum-level", type=int, default=0)
    parsers.env.add_argument("--extra-time", type=int, default=0)
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
