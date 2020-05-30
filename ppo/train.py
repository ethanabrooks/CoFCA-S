import abc
import functools
import itertools
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import dataset
import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from tensorboardX import SummaryWriter
from tqdm import tqdm

from common.atari_wrappers import wrap_deepmind
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo.agent import Agent, AgentValues
from ppo.control_flow.hdfstore import HDF5Store
from ppo.storage import RolloutStorage
from ppo.update import PPO
from ppo.utils import k_scalar_pairs, get_n_gpu, get_random_gpu
from ppo.wrappers import AddTimestep, TransposeImage, VecPyTorch, VecPyTorchFrameStack


# noinspection PyAttributeOutsideInit
class Train(abc.ABC):
    def setup(
        self,
        num_steps,
        num_processes,
        seed,
        cuda_deterministic,
        cuda,
        time_limit,
        gamma,
        normalize,
        log_interval,
        eval_interval,
        use_gae,
        tau,
        ppo_args,
        agent_args,
        render,
        render_eval,
        load_path,
        synchronous,
        num_batch,
        env_args,
        success_reward,
        use_tqdm,
    ):
        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        if render:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False

        # reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cuda &= torch.cuda.is_available()
        if cuda and cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)

        self.device = "cpu"
        if cuda:
            self.device = self.get_device()
        # print("Using device", self.device)

        self.envs = self.make_vec_envs(
            **env_args,
            seed=seed,
            gamma=(gamma if normalize else None),
            render=render,
            synchronous=True if render else synchronous,
            evaluation=False,
            num_processes=num_processes,
            time_limit=time_limit,
        )

        self.envs.to(self.device)
        self.agent = self.build_agent(envs=self.envs, **agent_args)
        self.rollouts = RolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_space=self.envs.observation_space,
            action_space=self.envs.action_space,
            recurrent_hidden_state_size=self.agent.recurrent_hidden_state_size,
            use_gae=use_gae,
            gamma=gamma,
            tau=tau,
        )

        # copy to device
        if cuda:
            tick = time.time()
            self.agent.to(self.device)
            self.rollouts.to(self.device)
            print("Values copied to GPU in", time.time() - tick, "seconds")

        self.ppo = PPO(agent=self.agent, num_batch=num_batch, **ppo_args)
        self.counter = Counter()

        self.i = 0
        if load_path:
            self._restore(load_path)

        self.make_train_iterator = lambda: self.train_generator(
            num_steps=num_steps,
            num_processes=num_processes,
            time_limit=time_limit,
            log_interval=log_interval,
            use_tqdm=use_tqdm,
            success_reward=success_reward,
        )
        self.train_iterator = self.make_train_iterator()

    def _train(self):
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = self.make_train_iterator()
            return self._train()

    def train_generator(
        self,
        num_steps,
        num_processes,
        time_limit,
        log_interval,
        success_reward,
        use_tqdm,
    ):
        self.envs.train()
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        tick = time.time()
        log_progress = None

        for _ in itertools.count(self.i):
            if self.i % log_interval == 0 and use_tqdm:
                log_progress = tqdm(total=log_interval, desc="next log")
            self.i += 1
            epoch_counter = self.run_epoch(
                obs=self.rollouts.obs[0],
                rnn_hxs=self.rollouts.recurrent_hidden_states[0],
                masks=self.rollouts.masks[0],
                num_steps=num_steps,
                counter=self.counter,
                success_reward=success_reward,
                use_tqdm=False,
            )

            with torch.no_grad():
                next_value = self.agent.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1],
                ).detach()

            self.rollouts.compute_returns(next_value=next_value)
            train_results = self.ppo.update(self.rollouts)
            self.rollouts.after_update()
            if log_progress is not None:
                log_progress.update()
            # print(self.i, self.i % self.log_interval)
            if self.i % log_interval == 0:
                total_num_steps = (self.i + 1) * num_processes * num_steps
                # print(f"Writing to {self.logdir}")
                fps = total_num_steps / (time.time() - tick)
                tick = time.time()
                yield dict(
                    k_scalar_pairs(tick=tick, fps=fps, **epoch_counter, **train_results)
                )

    def run_epoch(
        self, obs, rnn_hxs, masks, num_steps, counter, success_reward, use_tqdm
    ):
        # noinspection PyTypeChecker
        episode_counter = Counter(rewards=[], time_steps=[], success=[])
        iterator = range(num_steps)
        if use_tqdm:
            iterator = tqdm(iterator, desc="evaluating")
        for _ in iterator:
            with torch.no_grad():
                act = self.agent(
                    inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                )  # type: AgentValues

            # Observe reward and next obs
            obs, reward, done, infos = self.envs.step(act.action)

            for d in infos:
                for k, v in d.items():
                    episode_counter.update({k: float(v) / num_steps / len(infos)})

            # track rewards
            counter["reward"] += reward.numpy()
            counter["time_step"] += np.ones_like(done)
            episode_rewards = counter["reward"][done]
            episode_counter["rewards"] += list(episode_rewards)
            if success_reward is not None:
                # noinspection PyTypeChecker
                episode_counter["success"] += list(episode_rewards >= success_reward)

            episode_counter["time_steps"] += list(counter["time_step"][done])
            counter["reward"][done] = 0
            counter["time_step"][done] = 0

            # If done then clean the history of observations.
            masks = torch.tensor(
                1 - done, dtype=torch.float32, device=obs.device
            ).unsqueeze(1)
            rnn_hxs = act.rnn_hxs
            if self.rollouts is not None:
                self.rollouts.insert(
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
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    @staticmethod
    def make_env(env_id, seed, rank, add_timestep, time_limit, evaluation):
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, "atari") and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv
        )
        env.seed(seed + rank)
        obs_shape = env.observation_space.shape
        if add_timestep and len(obs_shape) == 1 and str(env).find("TimeLimit") > -1:
            env = AddTimestep(env)
        if is_atari and len(env.observation_space.shape) == 3:
            env = wrap_deepmind(env)

        # elif len(env.observation_space.shape) == 3:
        #     raise NotImplementedError(
        #         "CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        if time_limit is not None:
            env = TimeLimit(env, max_episode_steps=time_limit)

        return env

    def make_vec_envs(
        self,
        num_processes,
        gamma,
        render,
        synchronous,
        env_id,
        add_timestep,
        seed,
        evaluation,
        time_limit,
        num_frame_stack=None,
        **env_args,
    ):
        envs = [
            functools.partial(  # thunk
                self.make_env,
                rank=i,
                env_id=env_id,
                add_timestep=add_timestep,
                seed=seed,
                evaluation=evaluation,
                time_limit=time_limit,
                **env_args,
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

        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack)
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
