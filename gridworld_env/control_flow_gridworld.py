import abc
from collections import OrderedDict, namedtuple
import copy

from gym import spaces
import numpy as np

from dataclasses import dataclass
import gridworld_env
from gridworld_env import SubtasksGridworld

LineTypes = namedtuple(
    "LineTypes", "If Else EndIf While EndWhile Subtask", defaults=list(range(6))
)

L = LineTypes()


class Line(abc.ABC):
    @abc.abstractmethod
    def to_tuple(self):
        raise NotImplementedError


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


class ControlFlowGridworld(SubtasksGridworld):
    def __init__(self, *args, n_subtasks, **kwargs):
        self.n_encountered = n_subtasks
        super().__init__(*args, n_subtasks=n_subtasks * 2, **kwargs)
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
        )
        self.non_existing = None
        self.existing = None
        self.last_condition = None
        self.last_condition_passed = None

        world = self
        # noinspection PyArgumentList

        class Subtask(self.Subtask, Line):
            def to_tuple(self):
                return self + (0, 0)

        self.Subtask = Subtask

        @dataclass
        class If(Line):
            object: int

            def __str__(self):
                return f"if {world.object_types[self.object]}:"

            def to_tuple(self):
                return 0, 0, 0, L.If, self.object

        self.If = If

        @dataclass
        class While(Line):
            object: int

            def __str__(self):
                return f"while {world.object_types[self.object]}:"

            def to_tuple(self):
                return 0, 0, 0, L.While, self.object

        self.While = While

    def task_string(self):
        def helper():
            indent = ""
            for subtask in self.subtasks:
                if isinstance(subtask, (self.If, self.While, self.If)):
                    yield str(subtask)
                    indent = "    "
                elif isinstance(subtask, (EndIf, EndWhile)):
                    yield str(subtask)
                    indent = ""
                elif isinstance(subtask, Else):
                    yield str(subtask)
                else:
                    yield f"{indent}{subtask}"

        return "\n".join(helper())

    def get_observation(self):
        obs = super().get_observation()
        subtasks = np.pad(
            [s.to_tuple() for s in self.subtasks],
            [(0, self.n_subtasks - len(self.subtasks)), (0, 0)],
            "constant",
        )
        obs.update(subtasks=subtasks)
        for (k, s) in self.observation_space.spaces.items():
            assert s.contains(obs[k])
        return OrderedDict(obs)

    def subtasks_generator(self):
        object_types = np.arange(len(self.object_types))
        self.non_existing = non_existing = [self.np_random.choice(object_types)]
        self.existing = existing = list(set(object_types) - set(non_existing))
        active_control = None
        line_type = None
        # noinspection PyTypeChecker
        for i in range(self.n_encountered):
            if i == self.n_encountered - 1:
                line_type = self.Subtask
            elif i == self.n_encountered - 2:
                # must terminate control blocks on last line
                lines = {self.If: EndIf, Else: EndIf, self.While: EndWhile}
                line_type = lines.get(type(active_control), self.Subtask)
            elif i == self.n_encountered - 3:
                # cannot start new control block with only 2 lines left
                if isinstance(active_control, self.If):
                    line_type = self.np_random.choice([self.Subtask, EndIf])
                else:
                    line_type = self.Subtask
            elif line_type in [self.While, self.If, Else]:
                # must follow condition with subtask
                line_type = self.Subtask
            else:
                assert i < self.n_encountered - 3
                available_lines = {
                    self.If: [self.Subtask, Else, EndIf],
                    EndWhile: [self.Subtask, EndIf],
                    self.While: [self.Subtask, EndWhile],
                }
                line_type = self.np_random.choice(
                    available_lines.get(
                        type(active_control), [self.If, self.While, self.Subtask]
                    )
                )

            if line_type in (self.If, self.While):
                passing = self.np_random.rand() < 0.5
                if line_type is self.While and len(existing) <= 1:
                    passing = False
                active_control = line_type(
                    object=self.np_random.choice(
                        existing
                        if (passing or active_control is None)
                        else non_existing
                    )
                )
                yield active_control
            elif line_type is Else:
                active_control = Else()
                yield Else()
            elif line_type in (EndIf, EndWhile):
                if line_type is EndWhile and active_control.object in existing:
                    existing.remove(active_control.object)
                active_control = None
                yield line_type()

            elif line_type is self.Subtask:
                if isinstance(active_control, self.While):
                    yield self.Subtask(
                        interaction=self.np_random.choice(
                            self.irreversible_interactions
                        ),
                        count=0,
                        object=active_control.object,
                    )
                else:
                    yield self.Subtask(
                        interaction=self.np_random.choice(len(self.interactions)),
                        count=0,
                        object=self.np_random.choice(existing),
                    )
            else:
                raise RuntimeError

    def get_required_objects(self, subtasks):
        available = []
        i = 0
        executed = True
        loop_count = None
        condition = None

        while i < len(self.subtasks):
            line = self.subtasks[i]
            if isinstance(line, (self.If, self.While)):
                condition = line.object
                executed = line.object in self.existing
                if executed and line.object not in available:
                    available += [line.object]
                    yield line.object
                if isinstance(line, self.While):
                    loop_count = self.np_random.randint(1, 3)
            elif isinstance(line, self.Subtask):
                if executed:
                    if line.object not in available:
                        yield line.object
                        available.append(line.object)
                    if line.interaction in self.irreversible_interactions:
                        available.remove(line.object)
            elif isinstance(line, Else):
                executed = not executed
            elif isinstance(line, EndIf):
                executed = True
            elif isinstance(line, EndWhile):
                executed = True
                if loop_count > 0:
                    if condition not in available:
                        available += [condition]
                        yield condition
                    loop_count -= 1

            i = self.get_next_idx(i, existing=available)

    def get_next_subtask(self):
        i = (self.subtask_idx or 0) + 1
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
                while not isinstance(line, EndIf):
                    i += 1
            else:
                i += 1

        elif isinstance(line, EndWhile):
            print("is", self.last_condition, "in", existing)
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
