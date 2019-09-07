from collections import deque, namedtuple

import gym
import numpy as np
from gym.utils import seeding

Last = namedtuple("Last", "answer reward")
Actions = namedtuple("Actions", "answer done")


class Env(gym.Env):
    def __init__(self, height, width, time_limit, seed):
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.width = width
        self.height = height
        self.answer = None
        self.last = None
        self.start = None
        self.t = None
        self.map = np.zeros([height, width], dtype=int)
        self.one_hots = np.eye(4)
        self.action_space = gym.spaces.Discrete(height * width * 2)
        self.observation_space = gym.spaces.MultiDiscrete(
            2 * np.ones((4, *self.map.shape))
        )

    def get_paths(self):
        steps = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

        def get_surrounding(x1, x2):
            for s1, s2 in steps:
                y1, y2 = x1 + s1, x2 + s2
                if 0 <= y1 < self.height and 0 <= y2 < self.width:
                    yield y1, y2

        start = tuple([self.random.randint(0, d) for d in self.map.shape])
        paths = [[start, s] for s in get_surrounding(*start)]

        def check_surrounding(prev, x):
            for s in get_surrounding(*x):
                if s != prev:
                    for p in paths:
                        if s in p:
                            return False
            return True

        def extend(*path):
            *_, prev, current = path
            if current is None:
                return
            surrounding = list(get_surrounding(*current))
            self.random.shuffle(surrounding)
            for candidate in surrounding:
                if check_surrounding(prev=current, x=candidate):
                    return candidate  # return first candidate that passes test

        while True:
            paths_complete = True
            for path in paths:
                n = extend(*path)
                if n is not None:
                    paths_complete = False
                    path.append(n)
            if paths_complete:
                return start, paths

    def reset(self):
        # self.start, nodes, self.answer = self.create_map()
        self.last = None
        self.t = 0
        start, paths = self.get_paths()
        self.map[:] = 0
        for *path, last in paths:
            for node in path:
                self.map[node] = 1
            self.map[last] = 2
        self.map[start] = 3
        self.random.shuffle(paths)
        self.answer = paths[0][-1]
        for path in paths[1:]:
            obstruct = path[self.random.randint(low=1, high=len(path))]
            self.map[obstruct] = 0
        return self.get_observation()

    def step(self, action: int):
        self.t += 1
        if self.t > self.time_limit:
            return self.get_observation(), -1, True, {}
        action = Actions(answer=action % self.map.size, done=action // self.map.size)
        answer = (action.answer // self.width, action.answer % self.width)
        if action.done:
            r = 1 if answer == self.answer else -1
            self.last = Last(reward=r, answer=answer)
        else:
            r = 0
        return self.get_observation(), r, bool(action.done), {}

    def get_observation(self):
        obs = self.one_hots[self.map].transpose(2, 0, 1)
        assert self.observation_space.contains(obs)
        return obs

    def render(self, mode="human"):
        if self.last:
            print(self.last)
        # print(self.one_hots[self.map])
        print(self.map)
        print("answer:", self.answer)

    def create_map(self):
        steps = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        chosen = set()
        nodes = deque()
        start = self.random.randint(low=np.zeros(2), high=self.map.shape)
        nodes += [start]
        prev = {}
        length = 0
        answer = None
        while nodes:
            x1, x2 = nodes.pop()
            surrounding = [(x1 + step1, x2 + step2) for step1, step2 in steps]

            # check if x1, x2 is terminal
            if [s for s in surrounding if s in chosen and s != prev[x1, x2]]:
                if answer is None:
                    answer = prev[x1, x2]  # first terminating
                continue  # terminate branch

            chosen.add((x1, x2))

            # add in-bounds neighbors to the queue
            for s1, s2 in surrounding:
                if 0 <= s1 < self.height and 0 <= s2 < self.width:
                    prev[s1, s2] = x1, x2
                    nodes.append((s1, s2))

            length += 1

        return start, chosen, answer

    def train(self):
        pass


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--height", default=4, type=int)
    parser.add_argument("--width", default=4, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        a1, a2 = [int(s) for s in string.split()]
        return a1 * args["width"] + a2, True

    keyboard_control.run(Env(**args), action_fn=action_fn)
