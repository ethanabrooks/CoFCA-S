from collections import OrderedDict

import numpy as np
from gym import spaces

import env


class Env(env.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.observation_space.spaces["lines"]
        self.observation_space.spaces["line"] = spaces.MultiDiscrete(
            np.array(self.line_space)
        )

    def preprocess_obs(self, obs: OrderedDict):
        obs["line"] = obs["lines"][obs["ptr"]]
        super().preprocess_obs(obs)
