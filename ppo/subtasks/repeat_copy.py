from gym import Env
import gym
import numpy as np
from gym.utils import seeding


class RepeatCopy(Env):
    def __init__(self, n: int, cell_size: int, seed=0):
        self.h = h = cell_size
        self.n = n
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(
            low=np.zeros(h + 2), high=np.ones(h + 2)
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4 * h + 1), high=np.ones(4 * h + 1)
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

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for s1, s2 in sequence():
            obs = np.concatenate([h, np.array([s1, s2])], axis=-1)
            assert self.observation_space.contains(obs)
            action = yield obs, r, False, {}
            i, f, o, g, a = np.split(action, [self.h * i for i in range(1, 5)])
            i, f, o = map(sigmoid, [i, f, o])
            g = np.tanh(g)

            def render():
                print(seq)
                print("i", i)
                print("f", f)
                print("o", o)
                print("a", a)

            self._render = render
            c = f * c + i * g
            h = o * 1 / (1 + np.exp(-c))
            r = -np.abs(s2 - a.item())
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
        parser.add_argument("--n", type=int, required=True)
        parser.add_argument("--cell-size", type=int, required=True)
