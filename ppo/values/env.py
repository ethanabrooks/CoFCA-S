from collections import deque, namedtuple

import gym
import numpy as np
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Last = namedtuple("Last", "answer reward")
Obs = namedtuple("Obs", "rewards values")


class Env(gym.Env):
    def __init__(self, size, time_limit, max_reward, min_reward, seed):
        self.time_limit = time_limit
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.random, self.seed = seeding.np_random(seed)
        self.size = size
        self.dims = np.array([size, size])
        self.transitions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])
        self.rewards = None
        self.values = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict(
            rewards=gym.spaces.Box(low=min_reward, high=max_reward, shape=self.dims),
            values=gym.spaces.Box(
                low=min_reward * time_limit,
                high=max_reward * time_limit,
                shape=self.dims,
            ),
        )

    def reset(self):
        self.rewards = self.random.random_integers(
            low=self.min_reward, high=self.max_reward, size=self.dims
        )
        self.values = np.zeros_like(self.rewards)
        positions = np.stack(np.meshgrid(np.arange(self.size), np.arange(self.size))).T
        next_pos = tuple(
            (
                positions.reshape((*self.dims, 1, 2))
                + self.transitions.reshape((1, 1, -1, 2))
            )
            .clip(0, self.size - 1)
            .transpose(3, 0, 1, 2)
        )

        # value iteration
        for _ in range(self.time_limit):
            self.values = self.rewards + self.values[next_pos].max(axis=-1)
        return self.get_observation()

    def step(self, action: np.array):
        return self.get_observation(), 0, True, {}

    def get_observation(self):
        obs = Obs(rewards=self.rewards, values=self.values)._asdict()
        assert self.observation_space.contains(obs)
        return obs

    def render(self, mode="human"):
        print(self.rewards)
        input("pause")


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--size", default=4, type=int)
    parser.add_argument("--max-reward", default=5, type=int)
    parser.add_argument("--min-reward", default=-5, type=int)
    parser.add_argument("--time-limit", default=8, type=int)
    parser.add_argument("--planning-time", default=16, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        x, y = string.split()
        return int(x) * args["size"] + int(y)

    keyboard_control.run(Env(**args), action_fn=action_fn)
