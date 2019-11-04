import itertools
from collections import namedtuple, deque, defaultdict

import gym
import numpy as np
from gym.utils import seeding
from rl_utils import hierarchical_parse_args

from ppo import keyboard_control
from ppo.utils import RED, RESET

Obs = namedtuple("Obs", "roads open")


class Env(gym.Env):
    def __init__(self, seed, n_states, baseline, flip_prob):
        self.flip_prob = flip_prob
        self.baseline = baseline
        self.n_states = n_states
        self.random, self.seed = seeding.np_random(seed)
        self.transitions = None
        self.state = None
        self.goal = None
        self.open = None
        self.eye = np.eye(n_states)
        if baseline:
            self.observation_space = gym.spaces.MultiBinary(n_states ** 3)
        else:
            self.observation_space = gym.spaces.Dict(
                Obs(
                    roads=gym.spaces.MultiDiscrete(np.ones((n_states, n_states))),
                    open=gym.spaces.MultiBinary(n_states),
                )
            )

    def step(self, action):
        action = int(action)
        self.state = ((self.eye[self.state] @ self.transitions) * self.open)[action]
        if self.state == 0:
            return self.get_observation(), -1, True, {}
        self.open = np.abs(
            self.open - self.random.choice(1, size=self.n_states, p=self.flip_prob)
        )
        t = self.state == self.goal
        r = float(t) - 1
        return self.get_observation(), r, t, {}

    def reset(self):
        self.transitions = self.random.random_integers(
            0, 1, size=[self.n_states, self.n_states]
        )
        np.fill_diagonal(self.transitions, 1)
        self.open = self.random.random_integers(0, 1, self.n_states)
        path = self.choose_path()
        self.state = path[0]
        self.goal = path[-1]
        return self.get_observation()

    @staticmethod
    def floyd_warshall(graph):
        paths = np.empty_like(graph, object)
        paths[:] = []
        distances = (1 - graph) * np.inf + 1
        for k in range(len(graph)):
            distance_through_k = distances[:, k : k + 1] + distances[k : k + 1]
            update = distance_through_k < distances
            distances[update] = distance_through_k[update]
            paths[update] = (paths[:, k : k + 1] + paths[k : k + 1])[update]
            # for i, j in itertools.product(range(self.n_states), range(self.n_states)):
            #     if update[i, j]:
            #         paths[i, j] = paths[i, k] + paths[k, j]
        return paths, distances

    def choose_path(self):
        paths, distances = self.floyd_warshall(self.transitions)  # all shortest paths
        path, distance = max(
            zip(paths.flatten(), distances.flatten()), key=lambda p, d: d
        )  # longest shortest path
        return path

    def get_observation(self):
        obs = Obs(roads=self.transitions, open=self.open)
        if self.baseline:
            obs = np.concatenate([o.flatten() for o in obs])
        else:
            obs = obs._asdict()
        assert self.observation_space.contains(obs)

    def render(self, mode="human"):
        for i, row in enumerate(self.transitions):
            print(">" if i == self.state else " ", end="")
            for x, open_road in zip(row, self.open):
                print(RESET if open_road else RED, x, sep="", end="")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-states", default=6, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    args = hierarchical_parse_args(parser)
    keyboard_control.run(Env(**args), actions="".join(map(str, range(args["n_lines"]))))
