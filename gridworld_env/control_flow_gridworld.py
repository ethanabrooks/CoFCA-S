import abc
from collections import OrderedDict, namedtuple
import copy
from enum import Enum

from gym import spaces
import numpy as np

from dataclasses import dataclass
import gridworld_env
from gridworld_env import SubtasksGridworld

Obs = namedtuple("Obs", "base subtask subtasks ignore")
LineTypes = namedtuple(
    "LineTypes", "Subtask If Else EndIf While EndWhile", defaults=list(range(6))
)

L = LineTypes()

TaskTypes = Enum("TaskTypes", "Subtasks If Else While General")


class Line(abc.ABC):
    @abc.abstractmethod
    def to_tuple(self):
        raise NotImplementedError

    def replace_object(self, obj):
        return self


class Else(Line):
    def __str__(self):
        return "else:"

    def to_tuple(self):
        return 0, 0, 0, L.Else, 0


class EndIf(Line):
    def __str__(self):
        return "endif"

    def to_tuple(self):
        return 0, 0, 0, L.EndIf, 0


class EndWhile(Line):
    def __str__(self):
        return "endwhile"

    def to_tuple(self):
        return 0, 0, 0, L.EndWhile, 0


WHILE_PASSING_PROBS = [
    None,  # 0
    1,  # 1
    0.791288,  # 2
    0.745606,  # 3
    0.729545,  # 4
    0.722634,  # 5
    0.719304,  # 6
    0.717579,  # 7
    0.716641,  # 8
    0.716111,  # 9
]


class ControlFlowGridworld(SubtasksGridworld):
    def __init__(self, *args, n_subtasks, task_type, max_loops, **kwargs):
        self.task_type = TaskTypes[task_type]
        self.max_loops = max_loops
        self.n_encountered = n_subtasks
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        self.irreversible_interactions = [
            j for j, i in enumerate(self.interactions) if i in ("pick-up", "transform")
        ]

        self.required_objects = None
        obs_spaces = self.observation_space.spaces
        subtask_nvec = obs_spaces["subtasks"].nvec
        subtask_nvec = np.pad(
            subtask_nvec,
            [(0, 0), (0, 1)],
            "constant",
            constant_values=len(LineTypes._fields),
        )
        subtask_nvec = np.pad(
            subtask_nvec,
            [(0, 0), (0, 1)],
            "constant",
            constant_values=len(self.object_types) + 1,  # +1 for not-a-condition
        )

        # noinspection PyProtectedMember
        self.observation_space.spaces.update(
            subtask=spaces.Discrete(
                self.observation_space.spaces["subtask"].n + 1
            ),  # +1 for terminating control_flow
            subtasks=spaces.MultiDiscrete(subtask_nvec),
            ignore=spaces.MultiBinary(self.n_subtasks),
        )
        self.non_existing = None
        self.existing = None
        self.last_condition = None
        self.last_condition_passed = None

        world = self
        # noinspection PyArgumentList

        class Subtask(self.Subtask, Line):
            def to_tuple(self):
                return self + (L.Subtask, 0)

            def replace_object(self, obj):
                return self._replace(object=obj)

        self.Subtask = Subtask

        @dataclass
        class If(Line):
            object: int

            def __str__(self):
                return f"if {world.object_types[self.object]}:"

            def to_tuple(self):
                return 0, 0, 0, L.If, 1 + self.object

            def replace_object(self, obj):
                return If(obj)

        self.If = If

        @dataclass
        class While(Line):
            object: int

            def __str__(self):
                return f"while {world.object_types[self.object]}:"

            def to_tuple(self):
                return 0, 0, 0, L.While, 1 + self.object

            def replace_object(self, obj):
                return While(obj)

        self.While = While

    def task_string(self):
        def helper():
            indent = ""
            for i, subtask in enumerate(self.subtasks):
                if isinstance(subtask, self.Subtask):
                    yield f"{i}:{indent}{subtask}"
                else:
                    yield f"{i}:{subtask}"
                if isinstance(subtask, (self.If, Else, self.While)):
                    indent = "    "
                if isinstance(subtask, (EndIf, EndWhile)):
                    indent = ""

        return "\n".join(helper())

    def get_observation(self):
        obs = super().get_observation()
        subtasks = np.pad(
            [s.to_tuple() for s in self.subtasks],
            [(0, self.n_subtasks - len(self.subtasks)), (0, 0)],
            "constant",
        )
        idxs = np.arange(self.n_subtasks)
        obs.update(subtasks=subtasks, ignore=idxs >= len(self.subtasks))
        for (k, s) in self.observation_space.spaces.items():
            assert s.contains(obs[k])
        return OrderedDict(obs)

    # def subtasks_generator(self):
    #     active_control = None
    #     line_type = None
    #     interactions = list(range(len(self.interactions)))
    #     # noinspection PyTypeChecker
    #     for i in range(self.n_encountered):
    #         try:
    #             failing = not active_control.object
    #         except AttributeError:
    #             failing = False
    #         if line_type in [self.While, self.If, Else]:
    #             # must follow condition with subtask
    #             line_type = self.Subtask
    #         elif i == self.n_encountered - 1 or (
    #             i == self.n_encountered - 2 and failing
    #         ):
    #             # Terminate all active controls
    #             if isinstance(active_control, (self.If, Else)):
    #                 line_type = EndIf
    #             elif isinstance(active_control, self.While):
    #                 line_type = EndWhile
    #             else:
    #                 assert active_control is None
    #                 line_type = self.Subtask
    #
    #         else:
    #             # No need to terminate controls. No preceding condition
    #             line_types = {
    #                 self.If: [self.Subtask, EndIf],
    #                 Else: [self.Subtask, EndIf],
    #                 self.While: [self.Subtask, EndWhile],
    #             }
    #             defaults = [self.Subtask]
    #             if i <= self.n_encountered - 3:
    #                 # need at least 3 lines left for Else and While
    #                 line_types[self.If] += [Else]
    #                 defaults += [self.While, self.If]
    #             line_type = self.np_random.choice(
    #                 line_types.get(type(active_control), defaults)
    #             )
    #
    #         # instantiate lines
    #         if line_type in (self.If, self.While):
    #             active_control = line_type(None)
    #             yield active_control
    #         elif line_type is Else:
    #             active_control = Else()
    #             yield Else()
    #         elif line_type in (EndIf, EndWhile):
    #             active_control = None
    #             yield line_type()
    #
    #         elif line_type is self.Subtask:
    #             yield self.Subtask(
    #                 interaction=self.np_random.choice(
    #                     self.irreversible_interactions
    #                     if active_control
    #                     else interactions
    #                 ),
    #                 count=0,
    #                 object=None,
    #             )
    #         else:
    #             raise RuntimeError

    def subtasks_generator(self):
        assert self.n_subtasks == 8
        self.np_random.shuffle(self.irreversible_interactions)
        if self.np_random.rand() < 0:  # TODO
            yield self.If(None)
            yield self.Subtask(
                interaction=self.irreversible_interactions[0], count=0, object=None
            )
            yield Else()
            yield self.Subtask(
                interaction=self.irreversible_interactions[1], count=0, object=None
            )
            yield EndIf()
            yield self.If(None)
            yield self.Subtask(
                interaction=self.np_random.choice(len(self.interactions)),
                count=0,
                object=None,
            )
            yield EndIf()
        else:
            yield self.While(None)
            yield self.Subtask(
                interaction=self.irreversible_interactions[0], count=0, object=None
            )
            yield EndWhile()

    def get_required_objects(self, subtasks):
        available = []
        i = 0
        existing, non_existing = self.np_random.choice(
            len(self.object_types), size=2, replace=False
        )
        non_existing = {non_existing}
        object_types = {i for i, _ in enumerate(self.object_types)}
        n_loops = 0
        while i < len(self.subtasks):
            line = self.subtasks[i]
            if isinstance(line, (self.If, self.While)):
                passing = self.np_random.rand() < (
                    WHILE_PASSING_PROBS[self.max_loops]
                    if isinstance(line, self.While)
                    else 0.5
                )
                if passing:
                    obj = existing
                    if obj not in available:
                        available += [obj]
                        yield obj
                else:
                    obj = self.np_random.choice(list(non_existing))

                self.subtasks[i] = line.replace_object(obj)
            elif isinstance(line, EndWhile):
                n_loops += 1
                passing = self.np_random.rand() < 0.5 and n_loops < self.max_loops
                if passing:
                    assert existing not in available
                    available += [existing]
                    yield existing
                else:
                    non_existing.add(existing)
                    existing = self.np_random.choice(list(object_types - non_existing))
            elif isinstance(line, self.Subtask):
                n_loops += 1
                self.subtasks[i] = line.replace_object(existing)
                if existing not in available:
                    available.append(existing)
                    yield existing
                if line.interaction in self.irreversible_interactions:
                    available.remove(existing)

            i = self.get_next_idx(i, existing=available)

        for i in range(len(subtasks)):
            line = self.subtasks[i]
            if isinstance(line, self.Subtask) and line.object is None:
                self.subtasks[i] = line._replace(object=existing)

    def get_next_subtask(self):
        if self.subtask_idx is None:
            i = 0
        else:
            i = self.subtask_idx + 1
        while i < len(self.subtasks) and not isinstance(self.subtasks[i], self.Subtask):
            i = self.get_next_idx(i, existing=self.objects.values())
        return i

    def get_next_idx(self, i, existing):
        line = self.subtasks[i]
        if isinstance(line, (self.If, self.While)):
            self.last_condition = line.object
            self.last_condition_passed = line.object in existing
            if self.last_condition_passed:
                i += 1
            else:
                while not isinstance(self.subtasks[i], (EndWhile, Else, EndIf)):
                    i += 1
        elif isinstance(line, Else):
            if self.last_condition_passed:
                while not isinstance(self.subtasks[i], EndIf):
                    i += 1
            else:
                i += 1

        elif isinstance(line, EndWhile):
            if self.last_condition in existing:
                while not isinstance(self.subtasks[i], self.While):
                    i -= 1
            i += 1
        else:
            i += 1

        return i


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args("4x4SubtasksGridWorld-v0")
    del kwargs["class_"]
    del kwargs["max_episode_steps"]
    kwargs.update(n_subtasks=n_subtasks, max_task_count=1)
    env = ControlFlowGridworld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = "wsadeq"
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == "__main__":
    import argparse
    import gridworld_env.keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subtasks", type=int, default=5)
    main(**vars(parser.parse_args()))
