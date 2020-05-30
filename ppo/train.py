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

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from common.atari_wrappers import wrap_deepmind
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo.agent import Agent, AgentValues
from ppo.storage import RolloutStorage
from ppo.update import PPO
from ppo.utils import get_n_gpu, get_random_gpu
from ppo.wrappers import AddTimestep, TransposeImage, VecPyTorch, VecPyTorchFrameStack


# noinspection PyAttributeOutsideInit
class Train(abc.ABC):
    def run(
        self,
        agent_args: dict,
        cuda: bool,
        cuda_deterministic: bool,
        env_args: dict,
        load_path: Path,
        log_dir: Path,
        log_interval: int,
        num_processes: int,
        num_steps: int,
        ppo_args: dict,
        render: bool,
        rollouts_args: dict,
        run_id,
        save_interval: int,
        seed: int,
        use_tqdm: bool,
    ):

        if render:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False

        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        # reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda &= torch.cuda.is_available()
        if cuda and cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)

        writer = SummaryWriter(logdir=str(log_dir)) if log_dir else None

        if cuda:
            match = re.search("\d+$", run_id) if run_id else None
            if match:
                device_num = int(match.group()) % get_n_gpu()
            else:
                device_num = get_random_gpu()

            device = torch.device("cuda", device_num)
        else:
            device = "cpu"
            print("Using device", device)

        self.envs = self.make_vec_envs(
            **env_args, seed=seed, render=render, num_processes=num_processes,
        )
        self.agent = self.build_agent(envs=self.envs, **agent_args)
        self.rollouts = RolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_space=self.envs.observation_space,
            action_space=self.envs.action_space,
            recurrent_hidden_state_size=self.agent.recurrent_hidden_state_size,
            **rollouts_args,
        )
        self.ppo = PPO(agent=self.agent, **ppo_args)

        # copy to device
        if cuda:
            tick = time.time()
            self.agent.to(device)
            self.rollouts.to(device)
            self.envs.to(device)
            print("Values copied to GPU in", time.time() - tick, "seconds")

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        log_progress = None

        if load_path:
            start = self.restore(load_path=load_path, device=device,)
        else:
            start = 0

        prev_epoch = 0
        tick = time.time()
        for epoch in itertools.count(start):
            if epoch % log_interval == 0 and use_tqdm:
                log_progress = tqdm(total=log_interval, desc="next log")
            counter = defaultdict(list)
            for k, v in self.run_epoch(num_steps=num_steps, use_tqdm=False):
                counter[k] += v

            train_results = self.ppo.update(self.rollouts)
            self.rollouts.after_update()
            if log_progress is not None:
                log_progress.update()

            if epoch % log_interval == 0:

                def compute_global_step(e):
                    return (e + 1) * num_processes * num_steps

                global_step = compute_global_step(epoch)

                def tag_value_pairs():
                    for k, v in counter.items():
                        yield k, np.mean(v)
                    new_steps = global_step - compute_global_step(prev_epoch)
                    fps = new_steps / (time.time() - tick)
                    yield "fps", fps

                prev_epoch = epoch
                tick = time.time()

                if writer is not None:
                    for k, v in tag_value_pairs():
                        writer.add_scalar(k, v, global_step)

                if log_dir and epoch % save_interval == 0:
                    self.save(checkpoint_dir=log_dir, epoch=epoch)

    def run_epoch(self, num_steps, use_tqdm):
        obs = self.rollouts.obs[0]
        rnn_hxs = self.rollouts.recurrent_hidden_states[0]
        masks = self.rollouts.masks[0]

        # noinspection PyTypeChecker
        iterator = range(num_steps)
        if use_tqdm:
            iterator = tqdm(iterator, desc="evaluating")
        counter = Counter()
        for _ in iterator:
            with torch.no_grad():
                act = self.agent(
                    inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                )  # type: AgentValues

            # Observe reward and next obs
            obs, reward, done, infos = self.envs.step(act.action)

            for d in infos:
                for k, v in d.items():
                    yield k, [v]

            counter["cumulative_reward"] += reward.numpy()
            counter["time_step"] += done.astype(int)
            for k, v in counter.items():
                yield k, list(v[done])
                v[done] = 0
            yield "reward", list(reward.numpy())

            # If done then clean the history of observations.
            masks = torch.tensor(
                1 - done, dtype=torch.float32, device=obs.device
            ).unsqueeze(1)
            rnn_hxs = act.rnn_hxs
            self.rollouts.insert(
                obs=obs,
                recurrent_hidden_states=act.rnn_hxs,
                actions=act.action,
                action_log_probs=act.action_log_probs,
                values=act.value,
                rewards=reward,
                masks=masks,
            )
        with torch.no_grad():
            next_value = self.agent.get_value(
                self.rollouts.obs[-1],
                self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1],
            ).detach()

        self.rollouts.compute_returns(next_value=next_value)

    @staticmethod
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    def make_vec_envs(
        self,
        num_processes,
        # gamma,
        render,
        synchronous,
        num_frame_stack=None,
        **env_args,
    ):
        synchronous = synchronous or render
        envs = [
            functools.partial(self.make_env, rank=i, **env_args,)  # thunk
            for i in range(num_processes)
        ]

        envs = (
            DummyVecEnv(envs, render=render)
            if (len(envs) == 1 or sys.platform == "darwin" or synchronous)
            else SubprocVecEnv(envs)
        )

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

    @staticmethod
    def make_env(env_id, seed, rank, add_timestep):
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

        # if time_limit is not None:
        #     env = TimeLimit(env, max_episode_steps=time_limit)

        return env

    def save(self, epoch, checkpoint_dir):
        modules = dict(
            optimizer=self.ppo.optimizer, agent=self.agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(checkpoint_dir, f"checkpoint.pt")
        torch.save(dict(epoch=epoch, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")

    def restore(self, load_path, device):
        load_path = load_path
        state_dict = torch.load(load_path, map_location=device)
        self.agent.load_state_dict(state_dict["agent"])
        self.ppo.optimizer.load_state_dict(state_dict["optimizer"])
        print(f"Loaded parameters from {load_path}.")
        return state_dict.get("step", -1) + 1
