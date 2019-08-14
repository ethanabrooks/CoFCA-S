import argparse

import ray
import torch.nn as nn
from gym.wrappers import TimeLimit
from ray.tune import tune, Trainable
from ray.tune.result import TIME_TOTAL_S
from ray.tune.schedulers import AsyncHyperBandScheduler

import ppo.events
import ppo.train
from ppo.events import Agent


class Train(ppo.train.Train, Trainable):
    def _setup(self, config):
        def setup(
            entropy_coef,
            hidden_size,
            learning_rate,
            num_layers,
            ppo_epoch,
            ppo_args,
            agent_args,
            **kwargs,
        ):

            self.setup(
                **kwargs,
                ppo_args=dict(
                    learning_rate=learning_rate, ppo_epoch=ppo_epoch, **ppo_args
                ),
                agent_args=dict(
                    entropy_coef=entropy_coef,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    **agent_args,
                ),
            )

        setup(**config)

    @staticmethod
    def make_env(time_limit, seed, rank, evaluation, **kwargs):
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
        return Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=False,
            **agent_args,
        )


parser = argparse.ArgumentParser()
parser.add_argument("--redis-address")
parser.add_argument("--log-dir")
parser.add_argument("--run-id")
args = parser.parse_args()
ray.init(redis_address=args.redis_address)
config = dict(
    num_processes=300,
    eval_interval=100,
    time_limit=30,
    cuda_deterministic=True,
    cuda=True,
    gamma=0.99,
    normalize=False,
    use_gae=ray.tune.choice([True, False]),
    tau=0.95,
    ppo_args=dict(clip_param=0.2, value_loss_coef=0.5, eps=1e-5, max_grad_norm=0.5),
    agent_args=dict(recurrent=True, activation=nn.ReLU()),
    render=False,
    render_eval=False,
    load_path=None,
    success_reward=None,
    synchronous=False,
    env_args={},
    log_interval=10,
    num_batch=ray.tune.choice([1, 2]),
    entropy_coef=ray.tune.uniform(low=0.01, high=0.04),
    hidden_size=ray.tune.choice([32, 64, 128, 512]),
    learning_rate=ray.tune.uniform(low=0.0002, high=0.002),
    num_layers=ray.tune.choice([0, 1, 2]),
    num_steps=ray.tune.choice([16, 32, 64]),
    ppo_epoch=ray.tune.choice([2, 3, 4, 5]),
    seed=ray.tune.choice(list(range(10))),
)

tune.run(
    Train,
    config=config,
    resources_per_trial=dict(cpu=1, gpu=0.5),
    checkpoint_freq=1,
    reuse_actors=True,
    local_dir=args.log_dir,
    scheduler=AsyncHyperBandScheduler(
        time_attr=TIME_TOTAL_S,
        metric="eval_rewards",
        mode="max",
        grace_period=3600,
        max_t=43200,
    ),
)
