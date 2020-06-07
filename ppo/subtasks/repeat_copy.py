from gym import Env
import gym
import numpy as np
from gym.utils import seeding


class RepeatCopy(Env):
    def __init__(self, n: int, seed=0):
        self.n = n
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(low=np.zeros(2), high=np.ones(2))
        self.action_space = gym.spaces.Box(low=np.zeros(1), high=np.ones(1))

    def generator(self):
        sequence = self.random.random(self.n)
        r = 0

        def render():
            print(sequence)

        self._render = render
        for s in sequence:
            ______ = yield (s, 0), r, False, {}
        for s in sequence:
            a = yield (0, 1), r, False, {}

            def render():
                print(sequence)
                print(s, a)

            self._render = render
            r = -np.abs(s - a)
        yield (0, 1), r, True, {}

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
        parser.add_argument("--n", type=int)
