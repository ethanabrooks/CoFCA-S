import os

import gym
from gym import spaces
import numpy as np


def int_to_bin_array(n: int):
    string = np.binary_repr(n, 2)
    return np.array(list(map(int, string)))


class SimplePOMDP(gym.Env):
    max_episode_steps = int(os.environ.get("MAX_EPISODE_STEPS", 2))

    def __init__(self):
        super().__init__()
        self.truth = None
        self.turn = 0
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,))

    def step(self, action):
        s = int_to_bin_array(self.n_actions)
        last_turn = SimplePOMDP.max_episode_steps - 1
        if self.turn == last_turn:  # requires the use of memory
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

    def render(self, mode="human"):
        pass


def main():
    # noinspection PyUnresolvedReferences
    env = gym.make("POMDP-v0")
    env.seed(1)
    s = env.reset()
    while True:
        print("obs:", s)
        action = int(input("act:"))
        s, r, t, _ = env.step(action)
        if t:
            print("obs:", s)
            print("reward:", r)
            print("resetting")
            s = env.reset()
            print()


if __name__ == "__main__":
    main()
