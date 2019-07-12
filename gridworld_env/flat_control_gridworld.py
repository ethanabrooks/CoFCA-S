from collections import Counter, OrderedDict, namedtuple
from enum import Enum

from gym import spaces
import numpy as np

from dataclasses import dataclass
from gridworld_env import SubtasksGridWorld

Obs = namedtuple("Obs", "base subtask subtasks")


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
        self.n_lines = 2 * n_subtasks
        kwargs.update(n_subtasks=self.n_lines)
        super().__init__(*args, **kwargs)
        self.n_subtasks = n_subtasks
        obs_spaces = self.observation_space.spaces
        subtask_nvec = obs_spaces["subtasks"].nvec[0]
        self.lines = None
        self.required_objects = None
        # noinspection PyProtectedMember
        self.observation_space.spaces.update(
            subtask=spaces.Discrete(self.n_lines + 1),
            subtasks=spaces.MultiDiscrete(
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
        interactions = list(range(len(self.interactions)))
        irreversible_interactions = [
            j for j, i in enumerate(self.interactions) if i in ("pick-up", "transform")
        ]

        object_types = np.arange(len(self.object_types))
        non_existing = [self.np_random.choice(object_types)]
        existing = list(set(object_types) - set(non_existing))
        self.lines = []
        one_step = False

        # noinspection PyTypeChecker
        for i in range(self.n_subtasks):
            if not one_step and i == self.n_subtasks - 1:
                one_step = True
            else:
                one_step = self.np_random.rand() < 0.5
            subtask_obj = self.np_random.choice(existing)
            self.np_random.shuffle(irreversible_interactions)
            passing_interaction, failing_interaction = (
                irreversible_interactions
                if i == 1
                else self.np_random.choice(interactions, size=2)
            )
            if one_step:
                branching = self.np_random.rand() < 0.5
                if branching:
                    condition_obj = self.np_random.choice(existing)
                    self.lines += [
                        self.If(condition_obj),
                        self.Subtask(
                            interaction=passing_interaction, count=0, object=subtask_obj
                        ),
                    ]

                else:  # not branching but still one-step
                    subtask_interaction = self.np_random.choice(interactions)
                    self.lines += [
                        self.Subtask(
                            interaction=subtask_interaction, count=0, object=subtask_obj
                        )
                    ]
            else:  # two-step
                self.lines += [
                    self.If(self.np_random.choice(non_existing)),
                    self.Subtask(
                        interaction=failing_interaction, count=0, object=subtask_obj
                    ),
                ]

        available = []
        self.required_objects = []
        passing = True
        for line in self.lines:
            if isinstance(line, self.If):
                passing = line.obj in existing
                if passing and line.obj not in available:
                    available += [line.obj]
                    self.required_objects += [line.obj]
            if passing and isinstance(line, self.Subtask):
                if line.object not in available:
                    self.required_objects += [line.object]
                    if line.interaction not in irreversible_interactions:
                        available += [line.object]

        return super().reset()

    def get_required_objects(self, _):
        yield from self.required_objects

    def get_observation(self):
        obs = super().get_observation()

        def get_task():
            for line in self.lines:
                if isinstance(line, self.If):
                    yield 0, 0, 0, line.obj
                elif isinstance(line, self.Subtask):
                    yield line + (0,)
                else:
                    raise NotImplementedError

        task = np.vstack(list(get_task()))
        lines = np.pad(task, [(0, self.n_lines - len(task)), (0, 0)], "constant")

        obs.update(subtasks=lines)
        for (k, s) in self.observation_space.spaces.items():
            assert s.contains(obs[k])
        return OrderedDict(filter_for_obs(obs))

    @property
    def subtask(self):
        if self.subtask_idx >= len(self.lines):
            return None

        subtask = self.lines[self.subtask_idx]
        assert isinstance(subtask, self.Subtask)
        return subtask

    def get_next_subtask(self):
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
