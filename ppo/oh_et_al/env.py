from collections import namedtuple
from itertools import permutations, chain, tee
import numpy as np

import gym
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Subtask = namedtuple("Subtask", "interaction object")
Obs = namedtuple("Obs", "obs subtasks")


class Env(gym.Env):
    def __init__(
        self, seed, height, width, n_subtasks, n_objects, implement_lower_level=False
    ):
        self.implement_lower_level = implement_lower_level
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
        self.subtask_idx = None
        if implement_lower_level:
            self.action_space = gym.spaces.Discrete(n_subtasks)
        else:
            self.action_space = gym.spaces.Discrete(
                len(self.transitions) + len(self.interactions)
            )
        self.observation_space = gym.spaces.Dict(
            Obs(
                obs=gym.spaces.MultiDiscrete(
                    np.array([[[3 + len(self.object_types)]], [[2]]])
                    * np.ones((2, *self.dims))
                ),
                subtasks=gym.spaces.MultiDiscrete(
                    np.tile(
                        np.array([[len(self.interactions), len(self.object_types)]]),
                        (self.n_subtasks, 1),
                    )
                ),
            )._asdict()
        )

    def step(self, action: int):
        action = int(action)
        pos = tuple(self.pos)
        if self.implement_lower_level:
            if action != self.subtask_idx:
                return self.get_observation(), -1, True, {}
            subtask = self.subtasks[action]
            if pos in self.objects and self.objects[pos] == subtask.object:
                action = len(self.transitions) + subtask.interaction
            else:
                goals = [
                    pos for pos, obj in self.objects.items() if obj == subtask.object
                ]
                if not goals:
                    action = self.random.choice(len(self.transitions))
                else:
                    goals = np.array(goals)
                    new_pos = self.pos + self.transitions
                    vectors = np.expand_dims(goals, 0) - np.expand_dims(new_pos, 1)
                    distances = np.abs(vectors).sum(-1)
                    action = distances.min(1).argmin()
        t = r = 0
        if action < len(self.transitions):
            self.pos += self.transitions[action]
            self.pos = np.clip(self.pos, (0, 0), self.dims - 1)
            interaction = "visit"
        else:
            interaction = self.interactions[action - len(self.transitions)]
        if pos in self.objects:
            if self.subtasks[self.subtask_idx] == Subtask(
                interaction=self.interactions.index(interaction),
                object=self.objects[pos],
            ):
                self.subtask_idx += 1
                if self.subtask_idx == len(self.subtasks):
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
        self.subtasks = [
            Subtask(object=o, interaction=i)
            for o, i in zip(list(self.objects.values()), interactions)
        ]
        self.subtask_idx = 0
        return self.get_observation()

    def get_observation(self):
        top_down = np.zeros((2, *self.dims), dtype=int)
        for pos, obj in self.objects.items():
            top_down[(0, *pos)] = obj + 1
        top_down[(1, *self.pos)] = 1

        obs = Obs(obs=top_down, subtasks=self.subtasks)._asdict()
        assert self.observation_space.contains(obs)
        return obs

    def render(self, mode="human", pause=True):
        top_down = Obs(**self.get_observation()).obs[0]
        top_down = self.icons[top_down]
        for i, row in enumerate(top_down):
            print("|", end="")
            for j, x in enumerate(row):
                print(REVERSE + x + RESET if (i, j) == tuple(self.pos) else x, end="")
            print("|")
        for i, subtask in enumerate(self.subtasks):
            print(
                ">" if i == self.subtask_idx else " ",
                Subtask(
                    object=self.object_types[subtask.object],
                    interaction=self.interactions[subtask.interaction],
                ),
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
    parser.add_argument("--implement-lower-level", action="store_true")
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        characters = "0123456789" if args["implement_lower_level"] else "dswaeq"
        try:
            return characters.index(string)
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
