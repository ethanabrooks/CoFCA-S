import itertools

import gym
import numpy as np
from gym.utils import seeding


class Bandit(gym.Env):
    def __init__(
        self, n: int, time_limit: int, seed=0,
    ):
        self.n = n
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, n])
        )
        self.action_space = gym.spaces.Discrete(n)

    def generator(self):
        statistics = self.random.random((self.n, 2))
        best = statistics[:, 0].max()
        r = -1
        a = -1
        for t in itertools.count():

            def render():
                for i, stat in enumerate(statistics):
                    print(i, stat)
                print("action:", a)
                print("reward", r)

            self._render = render

            r = self.random.normal(*statistics[a])
            i = dict(regret=best - r)
            o = (r, a)
            a = yield o, r, t == self.time_limit, i

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _, = next(self.iterator)
        return s

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n", default=10, type=int)
        parser.add_argument("--time-limit", default=100, type=int)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Bandit.add_arguments(PARSER)
    # Bandit(**vars(PARSER.parse_args())).interact()
