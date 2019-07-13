from collections import Counter, OrderedDict, namedtuple
from dataclasses import dataclass

from gym import spaces
import numpy as np

import gridworld_env
from gridworld_env import SubtasksGridWorld

Obs = namedtuple("Obs", "base subtask subtasks next_subtask lines")


class Else:
    def __str__(self):
        return "else:"


class EndIf:
    def __str__(self):
        return "endif"


def filter_for_obs(d):
    return {k: v for k, v in d.items() if k in Obs._fields}


class FlatControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, n_subtasks, **kwargs):
        n_subtasks += 1
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        self.passing_prob = 0.5
        self.pred = None
        self.force_branching = False

        self.conditions = None
        self.control = None
        self.required_objects = None
        obs_spaces = self.observation_space.spaces
        subtask_nvec = obs_spaces["subtasks"].nvec[0]
        self.lines = None
        self.required_objects = None
        # noinspection PyProtectedMember
        self.n_lines = self.n_subtasks + self.n_subtasks // 2 - 1
        self.observation_space.spaces.update(
            subtask=spaces.Discrete(self.observation_space.spaces["subtask"].n + 1),
            lines=spaces.MultiDiscrete(
                np.tile(
                    np.pad(
                        subtask_nvec,
                        [0, 1],
                        "constant",
                        constant_values=1 + len(self.object_types),
                    ),
                    (self.n_lines, 1),
                )
            ),
        )
        self.observation_space.spaces = Obs(
            **filter_for_obs(self.observation_space.spaces)
        )._asdict()
        world = self

        @dataclass
        class If:
            obj: int

            def __str__(self):
                return f"if {world.object_types[self.obj]}:"

        self.If = If

    def task_string(self):
        lines = iter(self.lines)

        def helper():
            while True:
                line = next(lines, None)
                if line is None:
                    return
                if isinstance(line, self.Subtask):
                    yield str(line)
                elif isinstance(line, self.If):
                    yield str(line)
                    yield f"    {next(lines)}"

        return "\n".join(helper())

    def reset(self):
        one_step_episode = self.np_random.rand() < 0.5
        if one_step_episode:
            self.branching_episode = self.np_random.rand() < 0.5

            # agent has to take one step when either
            if self.branching_episode:
                # encontering a passing condition
                self.passing_prob = 1
            else:
                # or a subtask
                self.passing_prob = 0.5
        else:
            # agent has to take two steps when encountering a failed condition
            self.branching_episode = True
            self.passing_prob = 0

        self.control = np.minimum(
            1 + np.array(list(self.get_control())), self.n_subtasks
        )
        object_types = np.arange(len(self.object_types))
        existing = self.np_random.choice(
            object_types, size=len(self.object_types) // 2, replace=False
        )
        non_existing = np.array(list(set(object_types) - set(existing)))
        n_passing = self.np_random.choice(
            2, p=[1 - self.passing_prob, self.passing_prob], size=self.n_subtasks
        ).sum()
        passing = self.np_random.choice(existing, size=n_passing)
        failing = self.np_random.choice(non_existing, size=self.n_subtasks - n_passing)
        self.conditions = np.concatenate([passing, failing])
        self.np_random.shuffle(self.conditions)
        self.passing = self.conditions[0] in passing
        self.required_objects = passing
        self.pred = False
        self.subtasks = list(self.subtasks_generator())

        def get_lines():
            for subtask, (pos, neg), condition in zip(
                self.subtasks, self.control, self.conditions
            ):
                yield subtask
                if pos != neg:
                    yield self.If(condition)

        self.lines = list(get_lines())[1:]
        o = super().reset()
        self.subtask_idx = self.get_next_subtask()
        return o

    def step(self, action):
        s, r, t, i = super().step(action)
        i.update(passing=self.passing)
        return s, r, t, i

    def get_observation(self):
        obs = super().get_observation()

        def get_lines():
            for line in self.lines:
                if isinstance(line, self.Subtask):
                    yield line + (0,)
                elif isinstance(line, self.If):
                    yield (0, 0, 0) + (1 + line.obj,)
                else:
                    raise NotImplementedError

        lines = np.pad(
            list(get_lines()), [(0, self.n_lines - len(self.lines)), (0, 0)], "constant"
        )
        obs.update(lines=lines)
        for (k, s) in self.observation_space.spaces.items():
            assert s.contains(obs[k])
        return OrderedDict(filter_for_obs(obs))

    def get_control(self):
        for i in range(self.n_subtasks):
            if i % 3 == 0:
                if self.branching_episode:
                    yield i + 1, i
                else:
                    yield i, i
            elif i % 3 == 1:
                if self.branching_episode:
                    yield i + 2, i + 2  # terminate
                else:
                    yield i, i
            elif i % 3 == 2:
                yield i + 1, i + 1  # terminate

    def choose_subtasks(self):
        if not self.branching_episode:
            choices = self.np_random.choice(
                len(self.possible_subtasks), size=self.n_subtasks
            )
            for i in choices:
                yield self.Subtask(*self.possible_subtasks[i])
            return
        irreversible_interactions = [
            j for j, i in enumerate(self.interactions) if i in ("pick-up", "transform")
        ]
        passing, failing = irreversible_interactions
        conditional_object = self.np_random.choice(len(self.object_types))
        for i in range(self.n_subtasks):
            self.np_random.shuffle(irreversible_interactions)
            if i % 3 == 0:
                j = self.np_random.choice(len(self.possible_subtasks))
                yield self.Subtask(*self.possible_subtasks[j])
            if i % 3 == 1:
                yield self.Subtask(
                    interaction=passing, count=0, object=conditional_object
                )
            if i % 3 == 2:
                yield self.Subtask(
                    interaction=failing, count=0, object=conditional_object
                )

    # noinspection PyTypeChecker
    def subtasks_generator(self):
        subtasks = list(self.choose_subtasks())
        i = 0
        encountered = Counter(passing=[], failing=[], subtasks=[])
        while i < self.n_subtasks:
            condition = self.conditions[i]
            passing = condition in self.required_objects
            branching = self.control[i, 0] != self.control[i, 1]
            encountered.update(passing=[condition if branching and passing else None])
            encountered.update(
                failing=[condition if branching and not passing else None]
            )
            encountered.update(subtasks=[i])
            i = self.control[i, int(passing)]

        object_types = Counter(range(len(self.object_types)))
        self.required_objects = list(set(encountered["passing"]) - {None})
        available = Counter(self.required_objects)
        for l in encountered.values():
            l.reverse()

        for t, subtask_idx in enumerate(encountered["subtasks"]):
            subtask = subtasks[subtask_idx]
            obj = subtask.object
            to_be_removed = self.interactions[subtask.interaction] in {
                "pick-up",
                "transform",
            }

            def available_now():
                if to_be_removed:
                    required_for_future = Counter(set(encountered["passing"][t:]))
                    return available - required_for_future
                else:
                    return available

            while not available_now()[obj]:
                if to_be_removed:
                    prohibited = Counter(encountered["failing"][:t])
                else:
                    prohibited = Counter(encountered["failing"])
                if obj in prohibited:
                    obj = self.np_random.choice(list(object_types - prohibited))
                    subtasks[subtask_idx] = subtask._replace(object=obj)
                else:
                    available[obj] += 1
                    self.required_objects += [obj]

            if to_be_removed:
                available[obj] -= 1

        yield from subtasks

    def get_next_subtask(self):
        if self.subtask_idx is None:
            return 0
        if self.subtask_idx > self.n_subtasks:
            return None
        return self.control[self.subtask_idx, int(self.evaluate_condition())]

    def get_next_subtask2(self):
        if self.subtask_idx is None:
            i = 0
        else:
            i = self.subtask_idx + 1
        while True:
            if i >= len(self.lines):
                return i
            line = self.lines[i]
            if isinstance(line, self.Subtask):
                return i
            elif isinstance(line, self.If):
                if line.obj in self.objects.values():
                    i += 1
                else:
                    i += 2

    def evaluate_condition(self):
        self.pred = self.conditions[self.subtask_idx] in self.objects.values()
        return self.pred

    def get_required_objects(self, _):
        yield from self.required_objects


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args("4x4SubtasksGridWorld-v0")
    del kwargs["class_"]
    del kwargs["max_episode_steps"]
    kwargs.update(n_subtasks=n_subtasks, max_task_count=1)
    env = FlatControlFlowGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = "wsadeq"
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == "__main__":
    import argparse
    import gridworld_env.keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subtasks", type=int, default=5)
    main(**vars(parser.parse_args()))
