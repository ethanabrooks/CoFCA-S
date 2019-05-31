import gym
from gym import spaces
from rl_utils.numpy import vectorize, onehot
import numpy as np


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
                for o, n in zip(obs.reshape(len(obs), -1).T,
                                nvec.reshape(len(nvec), -1).T):
                    yield onehot(o, n)

            return np.concatenate(list(one_hots()), axis=-1)
