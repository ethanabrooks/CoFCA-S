from collections import deque, namedtuple

import gym
import numpy as np
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Last = namedtuple("Last", "answer reward")
Obs = namedtuple("Obs", "go rewards")


class Env(gym.Env):
    def __init__(self, size, time_limit, planning_time, max_reward, min_reward, seed):
        self.planning_time = planning_time
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
        self.answers = []

        d = max_reward - min_reward
        self.one_hots = np.eye(d, dtype=int)
        self.action_space = gym.spaces.Discrete(size ** 2)
        self.observation_space = gym.spaces.Dict(
            dict(
                rewards=gym.spaces.Box(
                    low=self.min_reward, high=self.max_reward, shape=self.dims
                ),
                go=gym.spaces.Discrete(2),
            )
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
        pos = tuple(self.pos)
        self.answers = [pos]
        # value iteration
        for _ in range(self.time_limit):
            values = self.rewards + values[next_pos].max(axis=-1)
        for _ in range(self.time_limit):
            transition = self.transitions[
                values[next_pos][self.answers[-1]].argmax(axis=-1)
            ]
            self.answers += [
                tuple(np.clip(self.answers[-1] + transition, 0, self.size - 1))
            ]
        return self.get_observation()

    def step(self, action: int):
        self.t += 1
        if self.t <= self.planning_time:
            return self.get_observation(), 0, False, {}
        answer = (action // self.size, action % self.size)
        r = float(answer == self.answers[self.t - self.planning_time])
        t = self.t == self.planning_time + self.time_limit
        return self.get_observation(), r, t, {}

    def get_observation(self):
        obs = Obs(rewards=self.rewards, go=int(self.t <= self.planning_time))._asdict()
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
