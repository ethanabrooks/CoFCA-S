import time

import gym
from gym import spaces
from gym.utils import seeding
from gym.utils.colorize import color2num
import numpy as np
import six
from gym.wrappers import TimeLimit


def int_to_bin_array(n: int):
    string = np.binary_repr(n, 2)
    return np.array(list(map(int, string)))


class LogicGridWorld(gym.Env):
    def __init__(self):
        super().__init__()
        self.truth = None
        self.turn = 0
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)

    def step(self, action):
        s = int_to_bin_array(self.n_actions)
        if self.turn > 0:  # requires the use of memory
            guess = int_to_bin_array(action)
            success = np.all(guess == self.truth)
            r = float(success)
            t = bool(success)
        else:
            r = 0
            t = False
        self.turn += 1
        return s, r, t, {}

    def reset(self):
        truth_int = np.random.randint(self.n_actions)
        self.truth = int_to_bin_array(truth_int)
        self.turn = 0
        return self.truth

    def render(self, mode='human'):
        pass


def main():
    env = TimeLimit(LogicGridWorld(), max_episode_steps=2)
    env.seed(1)
    s = env.reset()
    while True:
        print('obs:', s)
        action = int(input('act:'))
        s, r, t, _ = env.step(action)
        if t:
            print('obs:', s)
            print('reward:', r)
            print('resetting')
            s = env.reset()
            print()


if __name__ == '__main__':
    main()
