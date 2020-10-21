from contextlib import contextmanager

import gym
from gym import spaces
from gym.spaces import Box, Discrete
import numpy as np
import torch

from common.vec_env import VecEnvWrapper
from common.vec_env.vec_normalize import VecNormalize as VecNormalize_


class FlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.env.observation_space, Box):
            self.observation_space = Box(
                low=self.observation_space.low.flatten(),
                high=self.observation_space.high.flatten(),
            )
        else:
            raise NotImplementedError

    def observation(self, observation):
        return observation.flatten()


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
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
            dtype=self.observation_space.dtype,
        )

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
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = "cpu"
        # TODO: Fix data types
        self.action_bounds = (
            (torch.tensor(self.action_space.low), torch.tensor(self.action_space.high))
            if isinstance(self.action_space, Box)
            else None
        )

    def extract_numpy(self, obs):
        if isinstance(obs, dict):
            # print("VecPyTorch")
            # for k, x in obs.items():
            #     print(k, x.shape)
            return np.hstack([x.reshape(x.shape[0], -1) for x in obs.values()])
        elif not isinstance(obs, (list, tuple)):
            return obs
        assert len(obs) == 1
        return obs[0]

    def reset(self):
        obs = self.extract_numpy(self.venv.reset())
        return torch.from_numpy(obs).float().to(self.device)

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.extract_numpy(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def to(self, device):
        self.device = device
        self.venv.to(device)
        if self.action_bounds is not None:
            self.action_bounds = [t.to(device) for t in self.action_bounds]

    def evaluate(self):
        self.venv.evaluate()

    def train(self):
        self.venv.train()

    def increment_curriculum(self):
        self.venv.increment_curriculum()

    def preprocess(self, action):
        if self.action_bounds is not None:
            low, high = self.action_bounds
            action = torch.min(torch.max(action, low), high)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.squeeze(-1)
        return action


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, : -self.shape_dim0] = self.stacked_obs[:, self.shape_dim0 :]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()

    def to(self, device):
        self.stacked_obs = self.stacked_obs.to(device)
        self.venv.to(device)


class OneHotWrapper(gym.Wrapper):
    def wrap_observation(self, obs, observation_space=None):
        if observation_space is None:
            observation_space = self.observation_space
        if isinstance(observation_space, spaces.Discrete):
            return onehot(obs, observation_space.n)
        if isinstance(observation_space, spaces.MultiDiscrete):
            assert observation_space.contains(obs)

            def one_hots():
                nvec = observation_space.nvec
                for o, n in zip(
                    obs.reshape(len(obs), -1).T, nvec.reshape(len(nvec), -1).T
                ):
                    yield onehot(o, n)

            return np.concatenate(list(one_hots()), axis=-1)


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


class TupleActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete(
            np.array([space.n for space in env.action_space.spaces])
        )

    def action(self, action):
        return tuple(action)

    def reverse_action(self, action):
        return np.concatenate(action)
