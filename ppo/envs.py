# stdlib
# third party
import functools
import sys

import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn

from common.running_mean_std import RunningMeanStd
from common.vec_env import VecEnvWrapper
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from ppo.env_adapter import TasksDummyVecEnv, TasksSubprocVecEnv


def wrap_env(env_thunk,
             seed,
             rank,
             eval,
             add_timestep=False,
             max_episode_steps=None):
    env = env_thunk()
    is_atari = hasattr(gym.envs, 'atari') and isinstance(
        env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        raise NotImplementedError

    env.seed(seed + rank)
    if eval:
        env.set_task(rank)

    obs_shape = env.observation_space.shape

    if add_timestep and len(
            obs_shape) == 1 and str(env).find('TimeLimit') > -1:
        env = AddTimestep(env)

    if is_atari:
        if len(env.observation_space.shape) == 3:
            raise NotImplementedError
            # env = wrap_deepmind(env)
    elif len(env.observation_space.shape) == 3:
        raise NotImplementedError(
            "CNN models work only for atari,\n"
            "please use a custom wrapper for a custom pixel input env.\n"
            "See wrap_deepmind for an example.")

    # If the input has shape (W,H,3), wrap for PyTorch convolutions
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env)

    if max_episode_steps:
        assert not isinstance(env, TimeLimit)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_vec_envs(make_env,
                  seed,
                  num_processes,
                  gamma,
                  device,
                  normalize,
                  eval,
                  synchronous=False,
                  train_tasks=False,
                  num_frame_stack=None):
    envs = [
        functools.partial(make_env, eval=eval, seed=seed, rank=i)
        for i in range(num_processes)
    ]

    synchronous = synchronous or len(envs) == 1 or sys.platform == 'darwin'
    if train_tasks:
        if synchronous:
            envs = TasksDummyVecEnv(envs)
        else:
            envs = TasksSubprocVecEnv(envs)
    else:
        if synchronous:
            envs = DummyVecEnv(envs)
        else:
            envs = SubprocVecEnv(envs)

    if len(envs.observation_space.shape) == 1 and normalize:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)
    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskTask(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    @staticmethod
    def extract_numpy(obs):
        if not isinstance(obs, (list, tuple)):
            return obs
        assert len(obs) == 1
        return obs[0]

    def reset(self):
        obs = self.extract_numpy(self.venv.reset())
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.extract_numpy(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_, nn.Module):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def load_state_dict(self, state_dict, strict=True):
        ret = state_dict['ret']
        ret_rms = state_dict['ret_rms']
        assert isinstance(ret, np.ndarray)
        assert isinstance(ret_rms, (RunningMeanStd, type(None)))
        self.ret = ret
        self.ret_rms = ret_rms

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return dict(ret=self.ret, ret_rms=self.ret_rms)

    def forward(self, *input):
        raise NotImplementedError


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env
# /vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
