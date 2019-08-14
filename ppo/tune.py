import argparse

import ray
import torch.nn as nn
from gym.wrappers import TimeLimit
from ray.tune import tune, Trainable
from ray.tune.schedulers import PopulationBasedTraining

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

    def build_agent(self, envs, recurrent=None, device=None, **agent_args):
        return Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=False,
            **agent_args,
        )


parser = argparse.ArgumentParser()
parser.add_argument("--ray-redis-address")
args = parser.parse_args()
ray.init(redis_address=args.ray_redis_address)
hyperparams = dict(
    batch_size=[1, 10, 5],
    entropy_coef=ray.tune.uniform(low=0.01, high=0.04),
    hidden_size=[32, 64, 128, 512],
    learning_rate=ray.tune.uniform(low=0.0002, high=0.002),
    num_layers=[0, 1, 2],
    num_steps=[16, 32, 64],
    ppo_epoch=[2, 3, 4, 5],
    seed=list(range(10)),
)
config = dict(
    num_processes=300,
    eval_interval=100,
    time_limit=30,
    cuda_deterministic=True,
    cuda=True,
    gamma=0.99,
    normalize=False,
    use_gae=True,
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
)
for k, v in hyperparams.items():
    if type(v) is list:
        config[k] = ray.tune.choice(v)
    else:
        config[k] = v

tune.run(
    Train,
    config=config,
    checkpoint_freq=1,
    reuse_actors=True,
    scheduler=PopulationBasedTraining(
        metric="rewards", mode="max", hyperparam_mutations=hyperparams
    ),
)
