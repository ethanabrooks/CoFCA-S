from collections import OrderedDict, namedtuple

from gym import spaces
import numpy as np

from dataclasses import dataclass
import gridworld_env
from gridworld_env import SubtasksGridworld

line_types = "If Else EndIf While EndWhile Subtask".split()
LineTypes = namedtuple("LineTypes", line_types, defaults=list(range(len(line_types))))


class Else:
    def __str__(self):
        return "else:"


class EndIf:
    def __str__(self):
        return "endif"


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
            subtask_nvec, [(0, 0), (0, 1)], "constant", len(LineTypes._fields)
        )
        subtask_nvec = np.pad(
            subtask_nvec, [(0, 0), (0, 1)], "constant", len(self.object_types) + 2
        )  # +2 for not-a-condition and previous-condition-evaluation

        # noinspection PyProtectedMember
        self.observation_space.spaces.update(
            subtask=spaces.Discrete(
                self.observation_space.spaces["subtask"].n + 1
            ),  # +1 for terminating subtasks
            subtasks=spaces.MultiDiscrete(subtask_nvec),
        )
        self.non_existing = None
        self.existing = None
        world = self

        @dataclass
        class If:
            object: int

            def __str__(self):
                return f"if {world.object_types[self.object]}:"

        self.If = If

    def task_string(self):
        lines = iter(self.subtasks)

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

    def get_observation(self):
        obs = super().get_observation()

        def get_lines():
            for line in self.subtasks:
                if isinstance(line, self.Subtask):
                    yield line + (0,)
                elif isinstance(line, self.If):
                    yield (0, 0, 0) + (1 + line.object,)
                else:
                    raise NotImplementedError

        lines = np.pad(
            list(get_lines()),
            [(0, self.n_subtasks - len(self.subtasks)), (0, 0)],
            "constant",
        )
        obs.update(subtasks=lines)
        for (k, s) in self.observation_space.spaces.items():
            assert s.contains(obs[k])
        return OrderedDict(obs)

    def reset(self):
        object_types = np.arange(len(self.object_types))
        self.non_existing = [self.np_random.choice(object_types)]
        self.existing = list(set(object_types) - set(self.non_existing))
        return super().reset()

    def subtasks_generator(self):
        interactions = list(range(len(self.interactions)))

        # noinspection PyTypeChecker
        for i in range(self.n_encountered):
            one_step = self.np_random.rand() < 0.5 or i == self.n_encountered - 1
            subtask_obj = self.np_random.choice(self.existing)
            self.np_random.shuffle(self.irreversible_interactions)
            passing_interaction, failing_interaction = (
                self.irreversible_interactions
                if i == 1
                else self.np_random.choice(interactions, size=2)
            )
            if one_step:
                control_flow = self.np_random.choice([None, "if", "while"])
                if control_flow == "if":
                    condition_obj = self.np_random.choice(self.existing)
                    # noinspection PyArgumentList
                    yield self.If(condition_obj)
                    yield self.Subtask(
                        interaction=passing_interaction, count=0, object=subtask_obj
                    )

                else:  # not branching but still one-step
                    subtask_interaction = self.np_random.choice(interactions)
                    yield self.Subtask(
                        interaction=subtask_interaction, count=0, object=subtask_obj
                    )
            else:  # two-step
                # noinspection PyArgumentList
                yield self.If(self.np_random.choice(self.non_existing))
                yield self.Subtask(
                    interaction=failing_interaction, count=0, object=subtask_obj
                )

    def get_required_objects(self, subtasks):
        available = []
        passing = True
        for line in subtasks:
            if isinstance(line, self.If):
                passing = line.object in self.existing
                if passing and line.object not in available:
                    available += [line.object]
                    yield line.object
            elif isinstance(line, self.Subtask):
                if passing:
                    if line.object not in available:
                        yield line.object
                        available += [line.object]
                    if line.interaction in self.irreversible_interactions:
                        available.remove(line.object)
                else:
                    passing = True

    def get_next_subtask(self):
        if self.subtask_idx is None:
            i = 0
        else:
            i = self.subtask_idx + 1
        while True:
            if i >= len(self.subtasks):
                return i
            line = self.subtasks[i]
            if isinstance(line, self.Subtask):
                return i
            elif isinstance(line, self.If):
                if line.object in self.objects.values():
                    i += 1
                else:
                    i += 2


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
