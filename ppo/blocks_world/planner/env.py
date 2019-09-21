import copy
import itertools
from collections import namedtuple, deque

import gym
import numpy as np
from gym.utils import seeding

from ppo.blocks_world.constraints import SideBySide, Stacked

Last = namedtuple("Last", "action reward terminal go")
Curriculum = namedtuple("Curriculum", "n_blocks search_depth")


class Env(gym.Env):
    def __init__(
        self,
        n_cols: int,
        seed: int,
        curriculum_level: int,
        extra_time: int,
        planning_steps: int,
    ):
        self.extra_time = extra_time
        self.n_rows = self.n_cols = n_cols
        self.n_grids = n_cols ** 2
        self.random, self.seed = seeding.np_random(seed)
        self.columns = None
        self.constraints = None
        self.n_blocks = None
        self.search_depth = None
        self.final_state = None
        self.time_limit = planning_steps + extra_time
        self.last = None
        self.solved = None
        self.t = None
        self.int_to_tuple = [(0, 0)]
        self.int_to_tuple.extend(itertools.permutations(range(self.n_cols), 2))
        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.int_to_tuple)] * planning_steps
        )
        # self.action_space = gym.spaces.Discrete(len(self.int_to_tuple))
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array([7] * (self.n_rows * self.n_cols * 2))
        )

        def curriculum_generator():
            last_curriculum = Curriculum(
                n_blocks=[
                    self.n_rows * self.n_cols // 3,
                    self.n_rows * self.n_cols // 3,
                ],
                search_depth=[1, 1],
            )
            max_blocks = self.n_rows * self.n_cols * 2 // 3
            while True:
                yield copy.deepcopy(last_curriculum)
                n_blocks = last_curriculum.n_blocks[1] + 1
                last_curriculum.n_blocks[1] = min(max_blocks, n_blocks)
                yield copy.deepcopy(last_curriculum)
                search_depth = last_curriculum.search_depth[1] + 1
                last_curriculum.search_depth[1] = min(planning_steps, search_depth)
                yield copy.deepcopy(last_curriculum)
                n_blocks = last_curriculum.n_blocks[0] + 1
                last_curriculum.n_blocks[0] = min(max_blocks, n_blocks)
                yield copy.deepcopy(last_curriculum)
                search_depth = last_curriculum.search_depth[0] + 1
                last_curriculum.search_depth[0] = min(planning_steps, search_depth)
                yield copy.deepcopy(last_curriculum)

        self.curriculum_level = curriculum_level
        self.curriculum_iterator = curriculum_generator()
        for _ in range(curriculum_level + 1):
            self.curriculum = next(self.curriculum_iterator)

    def valid(self, _from, _to, columns=None):
        if columns is None:
            columns = self.columns
        return columns[_from] and len(columns[_to]) < self.n_rows

    def step(self, action: list):
        _from, _to = self.int_to_tuple[int(action[self.t])]
        self.t += 1
        if self.solved:
            return self.get_observation(), 0, self.t >= self.time_limit, {}
        if self.valid(_from, _to):
            self.columns[_to].append(self.columns[_from].pop())
        if tuple(map(tuple, self.columns)) == self.final_state:
            r = 1
            self.solved = True
        else:
            r = 0
        t = self.t == self.time_limit
        self.last = Last(action=(_from, _to), reward=r, terminal=t, go=0)
        i = dict(
            n_blocks=self.n_blocks,
            search_depth=self.search_depth,
            curriculum_level=self.curriculum_level,
            reward_plus_curriculum=r + self.curriculum_level,
        )
        return self.get_observation(), r, t, i

    def reset(self):
        self.last = None
        self.solved = False
        self.t = 0
        self.n_blocks = self.random.random_integers(*self.curriculum.n_blocks)
        self.search_depth = self.random.random_integers(*self.curriculum.search_depth)
        self.columns = [[] for _ in range(self.n_cols)]
        blocks = list(range(1, self.n_blocks + 1))
        self.random.shuffle(blocks)
        for block in blocks:
            self.random.shuffle(self.columns)
            column = next(c for c in self.columns if len(c) < self.n_rows)
            column.append(block)
        trajectory = self.plan(self.columns, self.search_depth)
        if not trajectory:
            return self.reset()
        self.final_state = trajectory[0]
        return self.get_observation()

    def plan(self, start, max_depth):
        depth = 0
        start = tuple(map(tuple, start))
        back = {start: None}
        queue = deque([(depth, start)])
        while depth < max_depth and queue:
            depth, src = queue.popleft()
            actions = list(itertools.permutations(range(self.n_rows), 2))
            self.random.shuffle(actions)
            for _from, _to in actions:
                if self.valid(_from, _to, src):
                    dst = list(map(list, src))
                    dst[_to].append(dst[_from].pop())
                    dst = tuple(map(tuple, dst))
                    if dst not in back:
                        back[dst] = src
                        queue += [(depth + 1, dst)]
        trajectory = [src]
        node = src
        while node != start:
            node = back[node]
            trajectory.append(node)
        return trajectory

    def get_observation(self):
        obs = [
            x for r in self.pad(self.columns) + self.pad(self.final_state) for x in r
        ]
        assert self.observation_space.contains(obs)
        return obs

    def pad(self, columns):
        return [list(c) + [0] * (self.n_rows - len(c)) for c in columns]

    def increment_curriculum(self):
        self.curriculum_level += 1
        self.curriculum = next(self.curriculum_iterator)

    def render(self, mode="human", pause=True):
        print()
        print("state")
        for row in reversed(list(itertools.zip_longest(*self.columns))):
            for x in row:
                print("{:3}".format(x or " "), end="")
            print()
        print("goal")
        for row in reversed(list(itertools.zip_longest(*self.final_state))):
            for x in row:
                print("{:3}".format(x or " "), end="")
            print()
        print("curriculum level", self.curriculum_level)
        print("search depth", self.search_depth)
        print(f"time step: {self.t}/{self.time_limit}")
        print(self.last)
        if pause:
            input("pause")


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-cols", default=3, type=int)
    parser.add_argument("--curriculum-level", default=0, type=int)
    parser.add_argument("--extra-time", default=6, type=int)
    args = hierarchical_parse_args(parser)
    int_to_tuple = list(itertools.permutations(range(args["n_cols"]), 2))

    def action_fn(string):
        try:
            _from, _to = string
            index = int_to_tuple.index((int(_from), int(_to)))
            return index
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
