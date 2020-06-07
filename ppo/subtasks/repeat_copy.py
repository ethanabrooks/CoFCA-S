from gym import Env
import gym
import numpy as np
from gym.utils import seeding


class RepeatCopy(Env):
    def __init__(self, n: int, cell_size: int, use_cell: bool, seed=0):
        self.use_cell = use_cell
        self.h = h = cell_size
        self.n = n
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        obs_size = 2
        action_size = 1
        if use_cell:
            obs_size += h
            action_size += 4 * h
        self.observation_space = gym.spaces.Box(
            low=np.zeros(obs_size), high=np.ones(obs_size)
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(action_size), high=np.ones(action_size)
        )

    def generator(self):
        r = 0
        h = np.zeros(self.h)
        c = np.zeros(self.h)
        seq = self.random.random(self.n)

        def sequence():
            for s in seq:
                yield 0, s, s
            for _ in seq:
                yield 1, 0, s

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for s1, s2, s3 in sequence():
            obs = np.array([s1, s2])
            obs = np.concatenate([h, obs], axis=-1) if self.use_cell else obs
            assert self.observation_space.contains(obs)
            a = yield obs, r, False, {}
            if self.use_cell:
                i, f, o, g, a = np.split(a, [self.h * i for i in range(1, 5)])
                i, f, o = map(sigmoid, [i, f, o])
                g = np.tanh(g)
                c = f * c + i * g
                h = o * 1 / (1 + np.exp(-c))

            r = s1 * -np.abs(s3 - a.item())

            def render():
                print(seq)
                if self.use_cell:
                    print("i", i)
                    print("f", f)
                    print("o", o)
                print("ob", obs)
                print("a", a)
                print("r", r)

            self._render = render
        yield (0, 1), r, True, {}

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _, = next(self.iterator)
        return s

    def render(self, mode="human", pause=True):
        if self._render is not None:
            self._render()
        if pause:
            input()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n", type=int, required=True)
        parser.add_argument("--cell-size", type=int, required=True)
        parser.add_argument("--use-cell", action="store_true")
