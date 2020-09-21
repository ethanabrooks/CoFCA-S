import itertools
import inspect
import os
import sys
from abc import ABC
from collections import namedtuple, defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional

import gym
import numpy as np
import ray
import torch
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorboardX import SummaryWriter

from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.util import set_seeds
from networks import Agent, AgentOutputs, MLPBase
from ppo import PPO
from rollouts import RolloutStorage
from utils import k_scalar_pairs, get_device
from wrappers import VecPyTorch, get_vec_normalize

EpochOutputs = namedtuple("EpochOutputs", "obs reward done infos act masks")


class Aggregator(ABC):
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, **values):
        for k, v in values.items():
            self.values[k].append(v)


class MeanAggregator(Aggregator):
    def items(self):
        for k, v in self.values.items():
            yield k, np.mean(v)


class EvalMeanAggregator(MeanAggregator):
    def items(self):
        for k, v in super().items():
            yield "eval_" + k, v


class Trainer(tune.Trainable):
    def __init__(self, *args, **kwargs):
        self.iterator = None
        self.agent = None
        self.ppo = None
        self.i = None
        self.device = None
        super().__init__(*args, **kwargs)

    def setup(self, config):
        config = self.structure_config(**config)
        self.iterator = self.gen(**config)

    def structure_config(self, **config):
        agent_args = {}
        rollouts_args = {}
        ppo_args = {}
        gen_args = {}
        for k, v in config.items():
            if k in inspect.signature(self.build_agent).parameters:
                agent_args[k] = v
            if k in inspect.signature(Agent.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(MLPBase.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(RolloutStorage.__init__).parameters:
                rollouts_args[k] = v
            if k in inspect.signature(PPO.__init__).parameters:
                ppo_args[k] = v
            if k in inspect.signature(self.gen).parameters or k not in (
                list(agent_args.keys())
                + list(rollouts_args.keys())
                + list(ppo_args.keys())
            ):
                gen_args[k] = v
        config = dict(
            agent_args=agent_args,
            rollouts_args=rollouts_args,
            ppo_args=ppo_args,
            **gen_args,
        )
        return config

    def step(self):
        return next(self.iterator)

    def save_checkpoint(self, tmp_checkpoint_dir):
        modules = dict(
            optimizer=self.ppo.optimizer, agent=self.agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(tmp_checkpoint_dir, f"checkpoint.pt")
        torch.save(dict(step=self.i, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(state_dict["agent"])
        self.ppo.optimizer.load_state_dict(state_dict["optimizer"])
        start = state_dict.get("step", -1) + 1
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {checkpoint_path}.")

    def loop(self):
        while True:
            yield self.step()

    def gen(
        self,
        agent_args: dict,
        cuda: bool,
        cuda_deterministic: bool,
        env_args: dict,
        env_id: str,
        log_interval: int,
        normalize: float,
        num_batch: int,
        num_processes: int,
        ppo_args: dict,
        render_eval: bool,
        rollouts_args: dict,
        seed: int,
        synchronous: bool,
        train_steps: int,
        eval_interval: int = None,
        eval_steps: int = None,
        no_eval: bool = False,
        load_path: Path = None,
        render: bool = False,
    ):
        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        def make_vec_envs(evaluation):
            def env_thunk(rank):
                return lambda: self.make_env(
                    rank=rank, evaluation=evaluation, **env_args
                )

            env_fns = [env_thunk(i) for i in range(num_processes)]
            use_dummy = len(env_fns) == 1 or sys.platform == "darwin" or synchronous
            return VecPyTorch(
                DummyVecEnv(env_fns, render=render)
                if use_dummy
                else SubprocVecEnv(env_fns)
            )

        def run_epoch(obs, rnn_hxs, masks, envs, num_steps):
            for _ in range(num_steps):
                with torch.no_grad():
                    act = agent(
                        inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                    )  # type: AgentOutputs

                action = envs.preprocess(act.action)
                # Observe reward and next obs
                obs, reward, done, infos = envs.step(action)

                # If done then clean the history of observations.
                masks = torch.tensor(
                    1 - done, dtype=torch.float32, device=obs.device
                ).unsqueeze(1)
                yield EpochOutputs(
                    obs=obs, reward=reward, done=done, infos=infos, act=act, masks=masks
                )

                rnn_hxs = act.rnn_hxs

        if render_eval and not render:
            eval_interval = 1
        if render or render_eval:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False
        cuda &= torch.cuda.is_available()

        # reproducibility
        set_seeds(cuda, cuda_deterministic, seed)

        if cuda:
            self.device = torch.device("cuda") if self.name else get_device(self.name)
        else:
            self.device = torch.device("cpu")
        print("Using device", self.device)

        train_envs = make_vec_envs(evaluation=False)
        try:
            train_envs.to(self.device)
            agent = self.build_agent(envs=train_envs, **agent_args)
            if load_path:
                self.load_checkpoint(load_path)
            rollouts = RolloutStorage(
                num_steps=train_steps,
                obs_space=train_envs.observation_space,
                action_space=train_envs.action_space,
                recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
                **rollouts_args,
            )

            # copy to device
            if cuda:
                agent.to(self.device)
                rollouts.to(self.device)

            ppo = PPO(agent=agent, **ppo_args)
            train_means = MeanAggregator()

            for i in itertools.count():
                eval_means = EvalMeanAggregator()
                if eval_interval and not no_eval and i % eval_interval == 0:
                    # vec_norm = get_vec_normalize(eval_envs)
                    # if vec_norm is not None:
                    #     vec_norm.eval()
                    #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                    # self.envs.evaluate()
                    eval_masks = torch.zeros(num_processes, 1, device=self.device)
                    eval_envs = make_vec_envs(evaluation=True)
                    eval_envs.to(self.device)
                    with agent.recurrent_module.evaluating(eval_envs.observation_space):
                        eval_recurrent_hidden_states = torch.zeros(
                            num_processes,
                            agent.recurrent_hidden_state_size,
                            device=self.device,
                        )

                        for epoch_output in run_epoch(
                            obs=eval_envs.reset(),
                            rnn_hxs=eval_recurrent_hidden_states,
                            masks=eval_masks,
                            envs=eval_envs,
                            num_steps=eval_steps,
                        ):
                            eval_means.update(reward=epoch_output.reward.mean().item())
                            for info in epoch_output.infos:
                                eval_means.update(**info)
                    eval_envs.close()

                rollouts.obs[0].copy_(train_envs.reset())

                for epoch_output in run_epoch(
                    obs=rollouts.obs[0],
                    rnn_hxs=rollouts.recurrent_hidden_states[0],
                    masks=rollouts.masks[0],
                    envs=train_envs,
                    num_steps=train_steps,
                ):
                    train_means.update(reward=epoch_output.reward.mean().item())
                    for info in epoch_output.infos:
                        train_means.update(**info)
                    rollouts.insert(
                        obs=epoch_output.obs,
                        recurrent_hidden_states=epoch_output.act.rnn_hxs,
                        actions=epoch_output.act.action,
                        action_log_probs=epoch_output.act.action_log_probs,
                        values=epoch_output.act.value,
                        rewards=epoch_output.reward,
                        masks=epoch_output.masks,
                    )

                with torch.no_grad():
                    next_value = agent.get_value(
                        rollouts.obs[-1],
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1],
                    )

                rollouts.compute_returns(next_value.detach())
                train_results = ppo.update(rollouts)
                rollouts.after_update()

                total_num_steps = num_processes * train_steps * i
                if total_num_steps % log_interval == 0:
                    yield dict(
                        **train_results,
                        **dict(train_means.items()),
                        **dict(eval_means.items()),
                    )
                    train_means = MeanAggregator()
        finally:
            train_envs.close()

    @staticmethod
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    @staticmethod
    def make_env(env_id, seed, rank, evaluation, **kwargs):
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        return env

    @classmethod
    def main(
        cls,
        gpus_per_trial,
        cpus_per_trial,
        log_dir,
        num_iterations,
        num_samples,
        name,
        config,
        save_interval=None,
        **kwargs,
    ):
        cls.name = name
        if config is None:
            config = dict()
        for k, v in kwargs.items():
            if k not in config or v is not None:
                config[k] = v

        if log_dir:
            print("Not using tune, because log_dir was specified")
            writer = SummaryWriter(logdir=str(log_dir))
            trainer = cls(config)
            for i, result in enumerate(trainer.loop()):
                pprint(result)
                if writer is not None:
                    for k, v in k_scalar_pairs(**result):
                        writer.add_scalar(k, v, i)
                if (
                    None not in (log_dir, save_interval)
                    and (i + 1) % save_interval == 0
                ):
                    print("steps until save:", save_interval - i)
                    trainer.save_checkpoint(Path(log_dir, "checkpoint.pt"))
        else:
            local_mode = num_samples is None
            ray.init(dashboard_host="127.0.0.1", local_mode=local_mode)

            resources_per_trial = dict(gpu=gpus_per_trial, cpu=cpus_per_trial)

            if local_mode:
                print("Using local mode because num_samples is None")
                kwargs = dict()
            else:
                kwargs = dict(
                    search_alg=HyperOptSearch(config, metric="eval_reward"),
                    num_samples=num_samples,
                )
            if num_iterations:
                kwargs.update(stop=dict(training_iteration=num_iterations))

            tune.run(
                cls,
                name=name,
                config=config,
                resources_per_trial=resources_per_trial,
                **kwargs,
            )
