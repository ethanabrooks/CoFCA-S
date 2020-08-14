import abc
import functools
import itertools
import os
import re
import sys
from collections import defaultdict, namedtuple, Counter
from pathlib import Path
from pprint import pprint
from typing import Dict

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
from networks import Agent, AgentOutputs
from ppo import PPO
from rollouts import RolloutStorage
from utils import k_scalar_pairs, get_n_gpu, get_random_gpu
from wrappers import VecPyTorch

EpochOutputs = namedtuple("EpochOutputs", "obs reward done infos act masks")


class TrainBase(tune.Trainable):
    def __init__(self, *args, **kwargs):
        self.iterator = None
        self.agent = None
        self.ppo = None
        self.i = None
        self.device = None
        super().__init__(*args, **kwargs)

    def _setup(self, config):
        self.iterator = self.gen(**config)

    def _train(self):
        return next(self.iterator)

    def _save(self, tmp_checkpoint_dir):
        modules = dict(
            optimizer=self.ppo.optimizer, agent=self.agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(tmp_checkpoint_dir, f"checkpoint.pt")
        torch.save(dict(step=self.i, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")

    def _restore(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.agent.load_state_dict(state_dict["agent"])
        self.ppo.optimizer.load_state_dict(state_dict["optimizer"])
        start = state_dict.get("step", -1) + 1
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {checkpoint}.")

    def loop(self):
        yield from self.iterator

    def gen(
        self,
        train_steps,
        eval_steps,
        num_processes,
        seed,
        cuda_deterministic,
        cuda,
        normalize,
        num_epochs,
        log_interval,
        eval_interval,
        no_eval,
        ppo_args,
        agent_args,
        render,
        render_eval,
        rollouts_args,
        synchronous,
        num_batch,
    ):
        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        class EpochCounter:
            def __init__(self):
                self.episode_rewards = []
                self.episode_time_steps = []
                self.rewards = np.zeros(num_processes)
                self.time_steps = np.zeros(num_processes)

            def update(self, reward, done):
                self.rewards += reward.numpy()
                self.time_steps += np.ones_like(done)
                self.episode_rewards += list(self.rewards[done])
                self.episode_time_steps += list(self.time_steps[done])
                self.rewards[done] = 0
                self.time_steps[done] = 0

            def reset(self):
                self.episode_rewards = []
                self.episode_time_steps = []

            def items(self, prefix=""):
                if self.episode_rewards:
                    yield prefix + "rewards", np.mean(self.episode_rewards)
                if self.episode_time_steps:
                    yield prefix + "time_steps", np.mean(self.episode_time_steps)

        def make_vec_envs(evaluation):
            def env_thunk(rank):
                return self.make_env(
                    seed=seed, rank=rank, evaluation=evaluation, env_id=env_id
                )

            env_fns = [lambda: env_thunk(i) for i in range(num_processes)]
            use_dummy = len(env_fns) == 1 or sys.platform == "darwin" or synchronous
            return VecPyTorch(
                DummyVecEnv(env_fns, render=render)
                if use_dummy
                else SubprocVecEnv(env_fns)
            )

        def run_epoch(obs, rnn_hxs, masks, envs, num_steps):
            episode_counter = defaultdict(list)
            for _ in range(num_steps):
                with torch.no_grad():
                    act = self.agent(
                        inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                    )  # type: AgentOutputs

                # Observe reward and next obs
                obs, reward, done, infos = envs.step(act.action)
                self.process_infos(episode_counter, done, infos, **act.log)

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

        # reproducibility
        set_seeds(cuda, cuda_deterministic, seed)

        self.device = "cpu"
        if cuda:
            self.device = self.get_device()
        print("Using device", self.device)

        self.envs = self.make_vec_envs(
            seed=seed,
            render=render,
            synchronous=True if render else synchronous,
            evaluation=False,
            num_processes=num_processes,
        )
        self.make_eval_envs = functools.partial(
            self.make_vec_envs,
            seed=seed,
            render=render,
            synchronous=True if render else synchronous,
            evaluation=True,
            num_processes=num_processes,
        )

        self.envs.to(self.device)
        self.agent = self.build_agent(envs=self.envs, **agent_args)
        self.rollouts = RolloutStorage(
            num_steps=train_steps,
            num_processes=num_processes,
            obs_space=self.envs.observation_space,
            action_space=self.envs.action_space,
            recurrent_hidden_state_size=self.agent.recurrent_hidden_state_size,
            **rollouts_args,
        )

        # copy to device
        if cuda:
            self.agent.to(self.device)
            self.rollouts.to(self.device)

        self.ppo = PPO(agent=self.agent, num_batch=num_batch, **ppo_args)
        self.counter = Counter()

        self.i = 0

        self.make_train_iterator = lambda: self.train_generator(
            num_steps=train_steps,
            num_processes=num_processes,
            eval_steps=eval_steps,
            log_interval=log_interval,
            eval_interval=eval_interval,
            no_eval=no_eval,
        )
        self.train_iterator = self.make_train_iterator()

    def train_generator(
        self, num_steps, num_processes, eval_steps, log_interval, eval_interval, no_eval
    ):
        if eval_interval and not no_eval:
            # vec_norm = get_vec_normalize(eval_envs)
            # if vec_norm is not None:
            #     vec_norm.eval()
            #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            # self.envs.evaluate()
            eval_masks = torch.zeros(num_processes, 1, device=self.device)
            eval_counter = Counter()
            envs = self.make_eval_envs()
            envs.to(self.device)
            with self.agent.recurrent_module.evaluating(envs.observation_space):
                eval_recurrent_hidden_states = torch.zeros(
                    num_processes,
                    self.agent.recurrent_hidden_state_size,
                    device=self.device,
                )

                eval_result = self.run_epoch(
                    obs=envs.reset(),
                    rnn_hxs=eval_recurrent_hidden_states,
                    masks=eval_masks,
                    num_steps=eval_steps,
                    counter=eval_counter,
                    rollouts=None,
                    envs=envs,
                )
            envs.close()
            eval_result = {f"eval_{k}": v for k, v in eval_result.items()}
        else:
            eval_result = {}
        # self.envs.train()
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        log_progress = None

        for _ in range(self.i % eval_interval, eval_interval):
            self.i += 1
            epoch_counter = self.run_epoch(
                obs=self.rollouts.obs[0],
                rnn_hxs=self.rollouts.recurrent_hidden_states[0],
                masks=self.rollouts.masks[0],
                num_steps=num_steps,
                counter=self.counter,
                rollouts=self.rollouts,
                envs=self.envs,
            )

            with torch.no_grad():
                next_value = self.agent.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1],
                )

            self.rollouts.compute_returns(next_value.detach())
            train_results = self.ppo.update(self.rollouts)
            self.rollouts.after_update()
            if log_progress is not None:
                log_progress.update()
            if self.i % log_interval == 0:
                yield dict(**epoch_counter, **train_results, **eval_result)

    def run_epoch(self, obs, rnn_hxs, masks, num_steps, counter, rollouts, envs):
        # noinspection PyTypeChecker
        episode_counter = defaultdict(list)
        iterator = range(num_steps)
        for _ in iterator:
            with torch.no_grad():
                act = self.agent(
                    inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                )  # type: AgentOutputs

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(act.action)
            self.process_infos(episode_counter, done, infos, **act.log)

            # track rewards
            counter["reward"] += reward.numpy()
            counter["time_step"] += np.ones_like(done)
            episode_rewards = counter["reward"][done]
            episode_counter["rewards"] += list(episode_rewards)

            episode_counter["time_steps"] += list(counter["time_step"][done])
            counter["reward"][done] = 0
            counter["time_step"][done] = 0

            # If done then clean the history of observations.
            masks = torch.tensor(
                1 - done, dtype=torch.float32, device=obs.device
            ).unsqueeze(1)
            rnn_hxs = act.rnn_hxs
            if rollouts is not None:
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=act.rnn_hxs,
                    actions=act.action,
                    action_log_probs=act.action_log_probs,
                    values=act.value,
                    rewards=reward,
                    masks=masks,
                )

        return dict(episode_counter)

    @staticmethod
    def process_infos(episode_counter, done, infos, **act_log):
        for d in infos:
            for k, v in d.items():
                episode_counter[k] += v if type(v) is list else [float(v)]
        for k, v in act_log.items():
            episode_counter[k] += v if type(v) is list else [float(v)]

    @staticmethod
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    @staticmethod
    def make_env(env_id, seed, rank, add_timestep, evaluation):
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, "atari") and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv
        )
        env.seed(seed + rank)
        obs_shape = env.observation_space.shape

        # elif len(env.observation_space.shape) == 3:
        #     raise NotImplementedError(
        #         "CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape

        return env

    def make_vec_envs(
        self,
        num_processes,
        render,
        synchronous,
        seed,
        evaluation,
        num_frame_stack=None,
        **env_args,
    ):
        envs = [
            functools.partial(  # thunk
                self.make_env, rank=i, seed=seed, evaluation=evaluation, **env_args
            )
            for i in range(num_processes)
        ]

        if len(envs) == 1 or sys.platform == "darwin" or synchronous:
            envs = DummyVecEnv(envs, render=render)
        else:
            envs = SubprocVecEnv(envs)

        # if (
        # envs.observation_space.shape
        # and len(envs.observation_space.shape) == 1
        # ):
        # if gamma is None:
        # envs = VecNormalize(envs, ret=False)
        # else:
        # envs = VecNormalize(envs, gamma=gamma)

        envs = VecPyTorch(envs)

        # elif len(envs.observation_space.shape) == 3:
        #     envs = VecPyTorchFrameStack(envs, 4, device)

        return envs

    def _save(self, checkpoint_dir):
        modules = dict(
            optimizer=self.ppo.optimizer, agent=self.agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(checkpoint_dir, f"checkpoint.pt")
        torch.save(dict(step=self.i, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")
        return str(save_path)

    def _restore(self, checkpoint):
        load_path = checkpoint
        state_dict = torch.load(load_path, map_location=self.device)
        self.agent.load_state_dict(state_dict["agent"])
        self.ppo.optimizer.load_state_dict(state_dict["optimizer"])
        self.i = state_dict.get("step", -1) + 1
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {load_path}.")

    @abc.abstractmethod
    def get_device(self):
        raise NotImplementedError


class Trainer(TrainBase):
    def log_result(self, result):
        for k, v in k_scalar_pairs(**result):
            self.writer.add_scalar(k, v, total_num_steps)

    def get_device(self):
        match = re.search("\d+$", self.name)
        if match:
            device_num = int(match.group()) % get_n_gpu()
        else:
            device_num = get_random_gpu()

        return torch.device("cuda", device_num)

    @classmethod
    def main(
        cls,
        gpus_per_trial,
        cpus_per_trial,
        log_dir,
        num_samples,
        name,
        config,
        **kwargs,
    ):
        cls.name = name
        if config is None:
            config = dict()
        for k, v in kwargs.items():
            if v is not None:
                config[k] = v

        if log_dir:
            print("Not using tune, because log_dir was specified")
            writer = SummaryWriter(logdir=str(log_dir))
            for _ in itertools.count():
                trainer = cls(config)
                for i, result in enumerate(trainer.make_train_iterator()):
                    if writer is not None:
                        for k, v in k_scalar_pairs(**result):
                            writer.add_scalar(k, v, i)
            # for i, report in enumerate(cls(config).loop()):
            #     pprint(report)
            #     for k, v in report.items():
            #         writer.add_scalar(k, v, i)
        else:
            local_mode = num_samples is None
            ray.init(dashboard_host="127.0.0.1", local_mode=local_mode)
            metric = "final_reward"

            resources_per_trial = dict(gpu=gpus_per_trial, cpu=cpus_per_trial)
            kwargs = dict()

            if local_mode:
                print("Using local mode because num_samples is None")
            else:
                kwargs = dict(
                    search_alg=HyperOptSearch(config, metric=metric),
                    num_samples=num_samples,
                )

            tune.run(
                cls,
                name=name,
                config=config,
                resources_per_trial=resources_per_trial,
                **kwargs,
            )
