import gym
import numpy as np
from gym.utils import seeding


class NonStationaryContextualBandit(gym.Env):
    def __init__(self, n):
        super().__init__()
        self.rewards = np.zeros(n)
        self.random = np.random.RandomState()
        self.random_seed = None
        self.n = n
        self.observation_space = gym.spaces.Box(high=np.inf, low=np.inf, shape=(n,))
        self.action_space = gym.spaces.Discrete(n)
        self.optimal_return = None

    def train(self):
        pass

    def seed(self, seed=None):
        self.random, self.random_seed = seeding.np_random(int(seed))
        return [seed]

    def step(self, action: int):
        r = self.rewards[int(action)]
        self.optimal_return = 0
        s = self.random.standard_normal(size=self.n)
        self.rewards += s
        self.optimal_return += self.rewards.max()
        return s, r, False, dict(optimal=self.optimal_return)

    def reset(self):
        s = self.random.standard_normal(size=self.n)
        self.rewards += s
        self.optimal_return = 0
        return s

    def render(self, mode="human"):
        print(self.rewards)
