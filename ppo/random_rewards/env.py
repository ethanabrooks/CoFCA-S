from collections import deque, namedtuple

import gym
import numpy as np
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Last = namedtuple("Last", "answer reward")
Actions = namedtuple("Actions", "answer done")


class Env(gym.Env):
    def __init__(self, size, time_limit, no_op_limit, max_reward, min_reward, seed):
        self.no_op_limit = no_op_limit
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.size = size
        self.dims = np.array([size, size])
        self.transitions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])

        # reset
        self.rewards = None
        self.start = None
        self.t = None
        self.no_op_count = None
        self.pos = None
        self.optimal = None
        self.cumulative = None

        self.one_hots = np.eye(4)
        self.action_space = gym.spaces.Dict(
            dict(a=gym.spaces.Discrete(5), p=gym.spaces.Discrete(size ** 2))
        )
        self.observation_space = gym.spaces.Box(
            low=min_reward, high=max_reward, shape=self.dims
        )

    def reset(self):
        self.t = 0
        self.no_op_count = 0
        self.cumulative = 0
        self.rewards = self.random.random_integers(
            low=self.min_reward, high=self.max_reward, size=self.dims
        )
        self.pos = self.random.randint(low=np.zeros(2), high=self.dims)
        values = np.zeros_like(self.rewards)
        self.optimal = [0]
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
            values = self.rewards + values[next_pos].max(axis=-1)
            self.optimal += [values[tuple(self.pos)]]
        return self.get_observation()

    def step(self, action: tuple):
        action = int(action[0])
        if action == len(self.transitions):
            self.no_op_count += 1
            t = self.no_op_count > self.no_op_limit
            r = -1 if t else 0
            return self.get_observation(), r, t, {}
        self.cumulative += self.rewards[tuple(self.pos)]
        self.t += 1
        t = self.t >= self.time_limit
        info = dict(regret=self.optimal[self.t] - self.cumulative) if t else {}
        self.pos += self.transitions[action]
        self.pos.clip(0, self.size - 1, out=self.pos)
        r = 0
        if t:
            r = 1 if self.optimal[self.t] == self.cumulative else -1
        return self.get_observation(), r, t, info

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
        print("Time:", self.t)
        print("Cumulative:", self.cumulative)
        print("Optimal:", self.optimal[self.t])


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
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return "sdwa ".index(string)
        except ValueError:
            return None

    keyboard_control.run(Env(**args), action_fn=action_fn)
