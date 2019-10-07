import itertools
from collections import namedtuple
from itertools import permutations, chain, tee
from typing import List

import numpy as np

import gym
from gym.utils import seeding

from ppo.utils import REVERSE, RESET

Subtask = namedtuple("Subtask", "interaction object")
Obs = namedtuple("Obs", "obs subtasks n_subtasks")
Actions = namedtuple("Actions", "action beta")
Last = namedtuple("Last", "reward terminal")
Curriculum = namedtuple("Curriculum", "subtask_low subtask_high")


class Env(gym.Env):
    def __init__(
        self, seed, height, width, n_subtasks, time_limit, implement_lower_level=False
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
        self.time_limit = time_limit
        self.n_subtasks = n_subtasks
        self.evaluating = False
        self.pos = None
        self.objects = None
        self.subtask = None
        self.subtasks = None
        self.subtask_idx = None
        self.agent_idx = None
        self.last = None
        self.t = None
        self.action_space = gym.spaces.Dict(
            Actions(
                action=gym.spaces.Discrete(
                    len(self.transitions) + len(self.interactions)
                ),
                beta=gym.spaces.Discrete(2),
            )._asdict()
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
                n_subtasks=gym.spaces.Discrete(self.n_subtasks + 1),
            )._asdict()
        )

        def tasks():
            trained_object_types = set()
            trained_interactions = set()
            all_object_types = set(range(len(self.object_types)))
            all_interactions = set(range(len(self.interactions)))
            while not all(
                (
                    trained_object_types == all_object_types,
                    trained_interactions == all_interactions,
                )
            ):
                task = self.random_task()
                interactions, object_types = zip(*task)
                trained_object_types |= set(object_types)
                trained_interactions |= set(interactions)
                yield task

        self.training_tasks = list(tasks())  # type: List[List[Subtask]]

    def random_task(self) -> List[Subtask]:
        object_types = self.random.choice(len(self.object_types), self.n_subtasks)
        interactions = self.random.choice(len(self.interactions), self.n_subtasks)
        return [
            Subtask(object=o, interaction=i) for o, i in zip(object_types, interactions)
        ]

    def step(self, action: tuple):
        self.t += 1
        n_subtasks = len(self.subtasks)
        if self.t >= self.time_limit:
            return (
                self.get_observation(),
                0,
                True,
                dict(n_subtasks=n_subtasks, reward_plus_n_subtasks=n_subtasks),
            )

        pos = tuple(self.pos)
        actions = Actions(*action)
        if self.implement_lower_level:
            self.agent_idx += int(actions.beta)
            self.agent_idx = min(self.agent_idx, n_subtasks - 1)
            agent_subtask = self.subtasks[self.agent_idx]
            if pos in self.objects and self.objects[pos] == agent_subtask.object:
                action = len(self.transitions) + agent_subtask.interaction
            else:
                goals = [
                    pos
                    for pos, obj in self.objects.items()
                    if obj == agent_subtask.object
                ]
                if not goals:
                    action = self.random.choice(len(self.transitions))
                else:
                    goals = np.array(goals)
                    new_pos = self.pos + self.transitions
                    vectors = np.expand_dims(goals, 0) - np.expand_dims(new_pos, 1)
                    distances = np.abs(vectors).sum(-1)
                    action = distances.min(1).argmin()
        else:
            action = int(actions.action)
        r = 0
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
                if self.subtask_idx == n_subtasks:
                    r = 1
            if interaction == "pick-up":
                del self.objects[pos]
            if interaction == "transform":
                self.objects[pos] = len(self.object_types)
        t = bool(r)
        i = dict(n_subtasks=n_subtasks)
        self.last = Last(reward=r, terminal=t)
        if t:
            i.update(reward_plus_n_subtasks=n_subtasks + r)
        return self.get_observation(), r, t, i

    def reset(self):
        self.t = 0
        squares = np.prod(self.dims)
        n_subtasks = min(self.n_subtasks, squares - 1)
        ints = self.random.choice(squares, n_subtasks + 1, replace=False)
        self.pos, *objects_pos = np.stack(np.unravel_index(ints, self.dims), axis=1)
        if self.evaluating:
            self.subtasks = self.random_task()
        else:
            self.subtasks = self.training_tasks[
                self.random.choice(len(self.training_tasks))
            ]  # type: List[Subtask]
        object_types = [subtask.object for subtask in self.subtasks]
        self.random.shuffle(object_types)
        self.objects = dict(zip(map(tuple, objects_pos), object_types))
        self.subtask_idx = 0
        self.agent_idx = 0
        self.last = None
        return self.get_observation()

    def get_observation(self):
        top_down = np.zeros((2, *self.dims), dtype=int)
        for pos, obj in self.objects.items():
            top_down[(0, *pos)] = obj + 1
        top_down[(1, *self.pos)] = 1

        subtasks = np.array(self.subtasks)
        obs = Obs(
            obs=top_down, subtasks=subtasks, n_subtasks=len(self.subtasks)
        )._asdict()
        for name, space, o in zip(
            Obs._fields, self.observation_space.spaces.values(), obs.values()
        ):
            assert space.contains(o)
        return obs

    def increment_curriculum(self):
        self.curriculum = next(self.iterator)

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
        print(self.last)
        if pause:
            input("pause")

    def evaluate(self):
        self.evaluating = True

    def train(self):
        self.evaluating = False


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
