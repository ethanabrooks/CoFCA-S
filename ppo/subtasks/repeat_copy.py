from gym import Env
import gym
import numpy as np
from gym.utils import seeding


class RepeatCopy(Env):
    def __init__(
        self, n: int, h: int, seed=0,
    ):
        self.h = h
        self.n = n
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(low=np.zeros(2), high=np.ones(2))
        self.action_space = gym.spaces.Box(
            low=np.array(([0] * 3 * h) + [-1] * h + 1),
            high=np.array(([1] * 3 * h) + [1] * h + 1),
        )

    def generator(self):
        r = 0
        h = np.zeros(self.h)
        c = np.zeros(self.h)
        seq = self.random.random(self.n)

        def sequence():
            for s in seq:
                yield 0, s
            for _ in seq:
                yield 1, 0

        for s in sequence():
            i, f, o, g, a = yield (h, *s), r, False, {}

            def render():
                print(seq)
                print("i", i)
                print("f", f)
                print("o", o)
                print("a", a)

            self._render = render
            c = f * c + i * g
            h = o * np.sigmoid(c)
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
        parser.add_argument("-n", type=int)
        parser.add_argument("-h", type=int)
