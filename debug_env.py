import copy
import typing
from collections import Counter, defaultdict
from dataclasses import astuple, dataclass
from typing import Union, Dict, Generator, Tuple, List, Optional
from pprint import pprint

import numpy as np

import env
from data_types import (
    ActionType,
    X,
    Action,
)
from data_types import (
    Resource,
    Building,
    CoordType,
    WorldObject,
    Movement,
    Worker,
    State,
    Line,
    BuildOrder,
    ActionStage,
    Command,
    Targets,
    WorkerAction,
    WORLD_SIZE,
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
class DebugActionStage(ActionStage):
    action1: DebugAction1 = None
    ptr: int = 0
    active: ActionType = DebugAction1

    @classmethod
    def classes(cls):
        yield DebugAction1

    def _component_classes(self):
        yield self.action1

    def worker(self) -> Worker:
        assert isinstance(self.action1.worker, Worker)
        return self.action1.worker

    def command(self) -> Command:
        if isinstance(self.action1.target, Building):
            assert isinstance(self.action2, DebugAction2)
        return self.action1.target.assignment(None)


@dataclass
class Env(env.Env):
    def building_allowed(
        self,
        building: Building,
        dependency: Optional[Building],
        building_positions: Dict[CoordType, Building],
        insufficient_resources: bool,
        positions: Dict[WorldObject, CoordType],
        assignment_location: CoordType,
    ) -> bool:
        built = self.get_buildings(building_positions)
        # print(fg("green"), building, dependency, built, RESET)
        return dependency in built + [None] and assignment_location not in [
            *building_positions
        ]


def main(debug_env: bool, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
