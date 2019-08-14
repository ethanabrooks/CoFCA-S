import argparse
import numpy as np

import ray
import torch.nn as nn
from gym.wrappers import TimeLimit
from ray.tune import tune, Trainable
from ray.tune.result import TIME_TOTAL_S
from ray.tune.schedulers import AsyncHyperBandScheduler
from rl_utils import hierarchical_parse_args

import ppo.events
import ppo.train
from ppo.arguments import get_parser_with_exp_args
from ppo.events import Agent


class Train(ppo.train.Train, Trainable):
    def _setup(self, config):
        import ipdb

        ipdb.set_trace()
        self.setup(**config)

    @staticmethod
    def make_env(time_limit, seed, rank, evaluation, **kwargs):
        for k in kwargs:
            print(f"{k},")
        for k in kwargs:
            print(f"{k}={k},")
        import ipdb

        ipdb.set_trace()
        env = ppo.events.Wrapper(
            n_active_subtasks=2,
            watch_baby_range=2,
            avoid_dog_range=2,
            door_time_limit=7,
            max_time_outside=15,
            subtasks=[
                "ComfortBaby",
                "MakeFire",
                "WatchBaby",
                "AvoidDog",
                "KillFlies",
                "AnswerDoor",
            ],
            held_out=[["ComfortBaby", "MakeFire"]],
            evaluation=evaluation,
            env=ppo.events.Gridworld(
                height=4,
                width=4,
                cook_time=2,
                time_to_heat_oven=3,
                doorbell_prob=0.05,
                mouse_prob=0.2,
                baby_prob=0.1,
                mess_prob=0.01,
                fly_prob=0.005,
                toward_cat_prob=0.5,
            ),
        )
        env = TimeLimit(max_episode_steps=time_limit, env=env)
        env.seed(seed + rank)
        return env

    def get_device(self):
        return "cuda"

    def build_agent(self, envs, recurrent=None, device=None, **agent_args):
        for k in agent_args:
            print(f"{k},")
        for k in agent_args:
            print(f"{k}={k},")
        import ipdb

        ipdb.set_trace()
        return Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=False,
            **agent_args,
        )


def main(redis_address, debug, log_dir, **config):
    ray.init(redis_address=redis_address, local_mode=debug)

    config.update(
        use_gae=ray.tune.choice([True, False]),
        # ppo_args=dict(clip_param=0.2, value_loss_coef=0.5, eps=1e-5, max_grad_norm=0.5),
        # agent_args=dict(recurrent=True, activation=nn.ReLU()),
        num_batch=ray.tune.choice([1, 2]),
        entropy_coef=ray.tune.uniform(low=0.01, high=0.04),
        hidden_size=ray.tune.choice([32, 64, 128, 512]),
        learning_rate=ray.tune.uniform(low=0.0002, high=0.0001),
        num_layers=ray.tune.choice([0, 1, 2]),
        num_steps=ray.tune.choice([16, 32, 64]),
        ppo_epoch=ray.tune.sample_from(
            lambda spec: int(
                0.0035 / spec.config.learning_rate * np.random.uniform(low=-2, high=2)
            )
        ),
        seed=ray.tune.choice(list(range(10))),
    )

    tune.run(
        Train,
        config=config,
        resources_per_trial=dict(cpu=1, gpu=0.5),
        checkpoint_freq=1,
        reuse_actors=True,
        num_samples=1 if debug else 100,
        local_dir=log_dir,
        scheduler=AsyncHyperBandScheduler(
            time_attr=TIME_TOTAL_S,
            metric="eval_rewards",
            mode="max",
            grace_period=3600,
            max_t=43200,
        ),
    )


if __name__ == "__main__":
    parser = get_parser_with_exp_args()
    parser.add_argument("--redis-address")
    main(**hierarchical_parse_args(parser))
