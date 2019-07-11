from collections import Counter, OrderedDict, namedtuple
from enum import Enum

from gym import spaces
import numpy as np

from dataclasses import dataclass
from gridworld_env import SubtasksGridWorld

Obs = namedtuple("Obs", "base subtask lines")


@dataclass
class If:
    obj: int

    def __str__(self):
        return f"if {self.obj}:"


class Else:
    def __str__(self):
        return "else:"


class EndIf:
    def __str__(self):
        return "endif"


def filter_for_obs(d):
    return {k: v for k, v in d.items() if k in Obs._fields}


class FlatControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, n_iterations, **kwargs):
        self.n_iterations = n_iterations
        super().__init__(*args, **kwargs)
        obs_spaces = self.observation_space.spaces
        subtask_nvec = obs_spaces["subtasks"].nvec[0]
        self.n_lines = 5 * n_iterations
        self.lines = None
        # noinspection PyProtectedMember
        self.observation_space.spaces.update(
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
            )
        )
        self.observation_space.spaces = Obs(
            **filter_for_obs(self.observation_space.spaces)
        )._asdict()

    def task_string(self):
        lines_iter = iter(self.lines)

        def helper(indent):
            line = next(lines_iter)
            if isinstance(line, If):
                yield f"{indent}{line}"
                yield from helper(indent + "    ")
            elif isinstance(line, Else):
                yield f"{indent.rstrip('    ')}{line}"
                yield from helper(indent)
            elif isinstance(line, EndIf):
                indent = indent.rstrip("    ")
                yield f"{indent}{line}"
                yield from helper(indent)

        return "\n".join(helper(""))

    def get_observation(self):
        obs = super().get_observation()
        obs.update(lines=self.lines)
        return Obs(**filter_for_obs(obs))._asdict()

    def reset(self):
        interactions = list(range(len(self.interactions)))
        irreversible_interactions = [
            j for j, i in enumerate(self.interactions) if i in ("pick-up", "transform")
        ]

        object_types = np.arange(len(self.object_types))
        non_existing = [self.np_random.choice(object_types)]
        existing = list(set(object_types) - set(non_existing))
        task_counter = Counter(lines=[], required=[], available=[])

        # noinspection PyTypeChecker
        def generate_task(i):
            if i == 0:
                return
            one_step = self.np_random.rand() < 0.5
            subtask_obj = self.np_random.choice(existing)
            if subtask_obj not in task_counter["available"]:
                task_counter.update(required=[subtask_obj])
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
                    if condition_obj not in task_counter["available"]:
                        task_counter.update(
                            available=[condition_obj], required=[condition_obj]
                        )
                    task_counter.update(
                        lines=[
                            If(condition_obj),
                            self.Subtask(
                                interaction=passing_interaction,
                                count=0,
                                object=subtask_obj,
                            ),
                        ]
                    )

                    # recurse
                    generate_task(i - 1)

                    task_counter.update(
                        lines=[
                            Else,
                            self.Subtask(
                                interaction=failing_interaction,
                                count=0,
                                object=subtask_obj,
                            ),
                        ]
                    )
                else:  # not branching but still one-step
                    subtask_interaction = self.np_random.choice(interactions)
                    if subtask_interaction not in irreversible_interactions:
                        task_counter.update(available=[subtask_obj])

                    task_counter.update(
                        lines=[
                            self.Subtask(
                                interaction=subtask_interaction,
                                count=0,
                                object=subtask_obj,
                            )
                        ]
                    )
            else:  # two-step
                task_counter.update(
                    lines=[
                        If(self.np_random.choice(non_existing)),
                        self.Subtask(
                            interaction=failing_interaction, count=0, object=subtask_obj
                        ),
                        Else,
                        self.Subtask(
                            interaction=passing_interaction, count=0, object=subtask_obj
                        ),
                        EndIf,
                    ]
                )

                # recurse
                generate_task(i - 1)

        generate_task(self.n_iterations)
        self.required_objects = task_counter["required"]
        self.lines = task_counter["lines"]
        return super().reset()

    def get_observation(self):
        obs = super().get_observation()

        def get_lines():
            for subtask, (pos, neg), condition in zip(
                self.subtasks, self.control, self.conditions
            ):
                yield subtask + (0,)
                if pos != neg:
                    yield (0, 0, 0, condition + 1)

        lines = np.vstack(list(get_lines())[1:])
        self.lines = np.pad(lines, [(0, self.n_lines - len(lines)), (0, 0)], "constant")

        obs.update(lines=self.lines)
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
            yield from super().choose_subtasks()
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
