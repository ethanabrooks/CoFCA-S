from collections import deque, namedtuple

import gym
import numpy as np
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Last = namedtuple("Last", "answer reward")
Actions = namedtuple("Actions", "answer done")


class Env(gym.Env):
    def __init__(self, size, time_limit, seed):
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.size = size
        self.dims = np.array([size, size])
        self.transitions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])

        # reset
        self.rewards = None
        self.start = None
        self.t = None
        self.pos = None
        self.optimal = None

        self.one_hots = np.eye(4)
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.dims)

    def reset(self):
        self.t = 0
        self.rewards = self.random.random_integers(low=-1, high=1, size=self.dims)
        self.pos = self.random.randint(low=np.zeros(2), high=self.dims)
        self.optimal = self.compute_values()[self.pos]
        return self.get_observation()

    def compute_values(self):
        values = self.rewards
        positions = np.stack(np.meshgrid(np.arange(self.size), np.arange(self.size))).T
        next_pos = tuple(
            (
                positions.reshape((*self.dims, 1, 2))
                + self.transitions.reshape((1, 1, -1, 2))
            )
            .clip(0, self.size - 1)
            .transpose(3, 0, 1, 2)
        )
        for _ in range(self.time_limit):
            values = self.rewards + values[next_pos].max(axis=-1)
        return values

    def step(self, action: int):
        self.t += 1
        if self.t > self.time_limit:
            return self.get_observation(), 0, True, {}

        self.pos += self.transitions[action]
        self.pos.clip(0, self.size - 1, out=self.pos)
        return self.get_observation(), self.rewards[self.pos], False, {}

    def get_observation(self):
        obs = self.rewards
        assert self.observation_space.contains(obs)
        return obs

    def render(self, mode="human"):
        for i, row in enumerate(self.rewards):
            for j, r in enumerate(row):
                agent_pos = (i, j) == tuple(self.pos)
                if agent_pos:
                    print(REVERSE, end="")
                print("{0: <2}".format(r), end="")
                if agent_pos:
                    print(RESET, end="")
            print()


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--size", default=4, type=int)
    parser.add_argument("--time-limit", default=8, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        return "sdwa ".index(string, None)

    keyboard_control.run(Env(**args), action_fn=action_fn)
