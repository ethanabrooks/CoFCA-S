import typing
from collections import Counter
from dataclasses import astuple, dataclass
from typing import Generator
from typing import Union, Dict, Tuple, List

import numpy as np

import env
from data_types import (
    ActionType,
    X,
    Worker,
    Assignment,
    CompoundAction,
    Action,
    Targets,
    WORLD_SIZE,
    Resource,
    Coord,
    Building,
)
from data_types import (
    WorldObject,
    Movement,
    State,
    Line,
    BuildOrder,
    WorkerAction,
    Nexus,
)


@dataclass(frozen=True)
class DebugAction(Action):
    def next(self) -> ActionType:
        if self.reset():
            return DebugAction1
        return self.next_if_not_reset()


@dataclass(frozen=True)
class DebugAction1(DebugAction):
    target: X
    worker: X

    def to_ints(self) -> Generator[int, None, None]:
        yield Targets.index(self.target)
        yield self.worker.value - 1

    @classmethod
    def num_values(cls) -> "DebugAction1":
        return cls(target=len(Targets), worker=len(Worker))

    def reset(self):
        return isinstance(self.target, Resource)

    def next_if_not_reset(self) -> ActionType:
        return DebugAction2

    @classmethod
    def parse(cls, a) -> "Action":
        parsed = super().parse(a)
        assert isinstance(parsed, DebugAction1)
        return cls(target=Targets[parsed.target], worker=Worker(parsed.worker + 1))


@dataclass(frozen=True)
class DebugAction2(DebugAction):
    i: X
    j: X

    def to_ints(self) -> Generator[int, None, None]:
        yield self.i
        yield self.j

    @classmethod
    def num_values(cls) -> "DebugAction2":
        return cls(i=WORLD_SIZE, j=WORLD_SIZE)

    def reset(self):
        return True

    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError

    @classmethod
    def parse(cls, a) -> "Action":
        parsed = super().parse(a)
        assert isinstance(parsed, DebugAction2)
        return cls(i=parsed.i, j=parsed.j)


@dataclass(frozen=True)
class DebugCompoundAction(CompoundAction):
    action1: DebugAction1 = None
    ptr: int = 0
    active: ActionType = DebugAction1

    @classmethod
    def classes(cls):
        yield DebugAction1

    def actions(self):
        yield self.action1

    def worker(self) -> Worker:
        assert isinstance(self.action1.worker, Worker)
        return self.action1.worker

    def assignment(self) -> Assignment:
        if isinstance(self.action1.target, Building):
            assert isinstance(self.action2, DebugAction2)
        return self.action1.target.assignment(None)


@dataclass
class Env(env.Env):
    def building_allowed(
        self,
        building: Building,
        dependency: typing.Optional[Building],
        building_positions: List[Coord],
        insufficient_resources: bool,
        positions: Dict[WorldObject, Coord],
        assignment_location: Coord,
    ) -> bool:
        return assignment_location not in building_positions and dependency in [
            *building_positions,
            None,
        ]


def main(debug_env: bool, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
