from collections import namedtuple
from itertools import permutations, chain, tee
import numpy as np

import gym
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Subtask = namedtuple("Subtask", "object interaction")


class Env(gym.Env):
    def __init__(self, seed, height, width, n_subtasks, n_objects):
        self.transitions = np.array(
            list(chain(permutations(range(2), 2), permutations(range(-1, 1), 2)))
        )
        self.random, self.seed = seeding.np_random(seed)
        self.dims = np.array([height, width])
        self.interactions = ["pick-up", "transform", "visit"]
        self.object_types = ["cat", "sheep", "pig", "greenbot"]
        self.icons = np.array([" "] + [s[0] for s in self.object_types] + ["i"])
        self.n_objects = n_objects
        self.n_subtasks = n_subtasks
        self.pos = None
        self.objects = None
        self.subtask = None
        self.subtasks = None

    def step(self, action: int):
        action = int(action)
        t = r = 0
        if action < len(self.transitions):
            self.pos += self.transitions[action]
            self.pos = np.clip(self.pos, (0, 0), self.dims - 1)
            interaction = "visit"
        else:
            interaction = self.interactions[action - len(self.transitions)]
        pos = tuple(self.pos)
        if pos in self.objects:
            if self.subtask == Subtask(
                interaction=self.interactions.index(interaction),
                object=self.objects[pos],
            ):
                try:
                    self.subtask = next(self.subtasks)
                except StopIteration:
                    t = r = 1
            if interaction == "pick-up":
                del self.objects[pos]
            if interaction == "transform":
                self.objects[pos] = len(self.object_types)
        return self.get_observation(), r, t, {}

    def reset(self):
        ints = self.random.choice(np.prod(self.dims), self.n_objects + 1, replace=False)
        self.pos, *objects = np.stack(np.unravel_index(ints, self.dims), axis=1)
        object_types = self.random.choice(len(self.object_types), len(objects))
        self.objects = dict(zip(map(tuple, objects), object_types))
        interactions = self.random.choice(len(self.interactions), len(objects))
        self.subtasks = (
            Subtask(object=o, interaction=i)
            for o, i in zip(list(self.objects.values()), interactions)
        )
        self.subtask = next(self.subtasks)
        return self.get_observation()

    def get_observation(self):
        top_down = np.zeros(self.dims, dtype=int)
        for pos, obj in self.objects.items():
            top_down[pos] = obj + 1
        return top_down

    def render(self, mode="human", pause=True):
        top_down = self.get_observation()
        top_down = self.icons[top_down]
        for i, row in enumerate(top_down):
            print("|", end="")
            for j, x in enumerate(row):
                print(REVERSE + x + RESET if (i, j) == tuple(self.pos) else x, end="")
            print("|")
        print(
            Subtask(
                object=self.object_types[self.subtask.object],
                interaction=self.interactions[self.subtask.interaction],
            )
        )
        if pause:
            input("pause")


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--height", default=4, type=int)
    parser.add_argument("--width", default=4, type=int)
    parser.add_argument("--n-subtasks", default=4, type=int)
    parser.add_argument("--n-objects", default=4, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return "dswaeq".index(string)
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
