import copy
import itertools
from collections import namedtuple

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
        n_constraints: int,
        planning_steps: int,
    ):
        self.n_constraints = n_constraints
        self.extra_time = extra_time
        self.n_rows = self.n_cols = n_cols
        self.n_grids = n_cols ** 2
        self.random, self.seed = seeding.np_random(seed)
        self.columns = None
        self.constraints = None
        self.n_blocks = None
        self.search_depth = None
        self.time_limit = None
        self.last = None
        self.solved = None
        self.t = None
        self.int_to_tuple = [(0, 0)]
        self.int_to_tuple.extend(itertools.permutations(range(self.n_cols), 2))
        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.int_to_tuple)] * (1 + planning_steps)
        )
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array([7] * (self.n_rows * self.n_cols + 3 * n_constraints))
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
                last_curriculum.search_depth[1] += 1
                yield copy.deepcopy(last_curriculum)
                n_blocks = last_curriculum.n_blocks[0] + 1
                last_curriculum.n_blocks[0] = min(max_blocks, n_blocks)
                yield copy.deepcopy(last_curriculum)
                last_curriculum.search_depth[0] += 1
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
        action, *_ = action
        self.t += 1
        if self.solved or self.t > self.time_limit:
            return (
                self.get_observation(),
                float(self.solved),
                self.t > self.time_limit,
                {},
            )
        _from, _to = self.int_to_tuple[int(action)]
        if self.valid(_from, _to):
            self.columns[_to].append(self.columns[_from].pop())
        satisfied = [c.satisfied(self.pad(self.columns)) for c in self.constraints]
        if all(satisfied):
            r = 1
            self.solved = True
        else:
            r = 0
        t = False
        self.last = Last(action=(_from, _to), reward=r, terminal=t, go=0)
        i = dict(
            n_blocks=self.n_blocks,
            search_depth=self.search_depth,
            curriculum_level=self.curriculum_level,
            reward_plus_curriculum=r + self.curriculum_level,
        )
        if self.solved:
            i.update(n_satisfied=np.mean(satisfied))
        return self.get_observation(), r, t, i

    def reset(self):
        self.last = None
        self.solved = False
        self.t = 0
        self.n_blocks = self.random.random_integers(*self.curriculum.n_blocks)
        self.search_depth = self.random.random_integers(*self.curriculum.search_depth)
        self.time_limit = self.search_depth + self.extra_time
        self.columns = [[] for _ in range(self.n_cols)]
        blocks = list(range(1, self.n_blocks + 1))
        self.random.shuffle(blocks)
        for block in blocks:
            self.random.shuffle(self.columns)
            column = next(c for c in self.columns if len(c) < self.n_rows)
            column.append(block)
        ahead = self.search_ahead([], self.columns, self.search_depth)
        if ahead is None:
            return self.reset()
        start_state = self.pad(self.columns)
        final_state = self.pad(ahead)

        def generate_constraints():
            for column in final_state:
                for bottom, top in itertools.zip_longest([0] + column, column + [0]):
                    yield Stacked(top, bottom)
            for row in itertools.zip_longest(*final_state):
                for left, right in zip((0,) + row, row + (0,)):
                    yield SideBySide(left, right)

        constraints = list(generate_constraints())
        self.constraints = [
            c
            for c in constraints
            if not c.satisfied(start_state) and c.satisfied(final_state)
        ]
        self.random.shuffle(self.constraints)
        self.constraints = self.constraints[: self.n_constraints]
        return self.get_observation()

    def search_ahead(self, trajectory, columns, n_steps):
        if n_steps == 0:
            return columns
        trajectory = trajectory + [tuple(map(tuple, columns))]
        actions = list(itertools.permutations(range(self.n_rows), 2))
        self.random.shuffle(actions)
        for _from, _to in actions:
            if self.valid(_from, _to, columns):
                columns = copy.deepcopy(columns)
                columns[_to].append(columns[_from].pop())
                if tuple(map(tuple, columns)) not in trajectory:
                    future_state = self.search_ahead(trajectory, columns, n_steps - 1)
                    if future_state is not None:
                        return future_state

    def get_observation(self):
        state = [c + [0] * (self.n_rows - len(c)) for c in self.columns]
        padding = [[0] * 3 * (self.n_constraints - len(self.constraints))]
        constraints = [c.list() for c in self.constraints] + padding
        obs = [x for r in state + constraints for x in r]
        assert self.observation_space.contains(obs)
        return obs

    def pad(self, columns):
        return [list(c) + [0] * (self.n_rows - len(c)) for c in columns]

    def increment_curriculum(self):
        pass
        # self.curriculum_level += 1
        # self.curriculum = next(self.curriculum_iterator)

    def render(self, mode="human", pause=True):
        print()
        for row in reversed(list(itertools.zip_longest(*self.columns))):
            for x in row:
                print("{:3}".format(x or " "), end="")
            print()
        for constraint in self.constraints:
            print(
                "{:3}".format("✔︎")
                if constraint.satisfied(self.pad(self.columns))
                else "  ",
                end="",
            )
            print(str(constraint))
        print("curriculum level", self.curriculum_level)
        print("search depth", self.search_depth)
        print(f"time step: {self.t}/{self.time_limit}")
        print(self.last)
        if pause:
            input("pause")

    def plan(self, trajectory, action_list):
        columns = list(map(list, trajectory[-1]))
        if tuple(map(tuple, columns)) in trajectory[:-1]:
            return
        if all(c.satisfied(self.pad(columns)) for c in self.constraints):
            return action_list
        actions = list(itertools.permutations(range(self.n_rows), 2))
        self.random.shuffle(actions)
        for _from, _to in actions:
            if self.valid(_from, _to, columns):

                new_columns = copy.deepcopy(columns)
                new_columns[_to].append(new_columns[_from].pop())

                plan = self.plan(
                    trajectory=trajectory + [tuple(map(tuple, new_columns))],
                    action_list=action_list + [(_from, _to)],
                )
                if plan is not None:
                    return plan


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
