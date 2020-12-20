import itertools
import os
import typing
from abc import abstractmethod, ABC
from collections import Counter
from dataclasses import dataclass, astuple, fields, replace
from enum import unique, Enum, auto
from typing import Tuple, Union, List, Generator, Dict, Generic, Optional

import numpy as np
import torch
from colored import fg
from gym import Space

from utils import RESET, Discrete

Coord = Tuple[int, int]

WORLD_SIZE = None


class WorldObject:
    @property
    @abstractmethod
    def symbol(self):
        return

    @abstractmethod
    def __eq__(self, other):
        pass


@unique
class Worker(WorldObject, Enum):
    A = auto()
    B = auto()
    C = auto()

    @property
    def symbol(self):
        return self.value

    def __eq__(self, other):
        # noinspection PyArgumentList
        return Enum.__eq__(self, other)

    def __hash__(self):
        # noinspection PyArgumentList
        return Enum.__hash__(self)


class Assignment:
    @abstractmethod
    def action(
        self,
        current_position: Coord,
        positions: Dict[Union["Resource", Worker], Coord],
        nexus_positions: List[Coord],
    ) -> "WorkerAction":
        raise NotImplementedError


class Target:
    @abstractmethod
    def assignment(self, action3: Optional["IJAction"]) -> "Assignment":
        raise NotImplementedError

    @classmethod
    def index(cls, item: "Target") -> int:
        return [*cls].index(item)


class WorkerAction:
    pass


@unique
class Resource(WorldObject, Target, Assignment, Enum):
    MINERALS = auto()
    GAS = auto()

    def __hash__(self):
        return Enum.__hash__(self)

    def assignment(self, action3: Optional["IJAction"]) -> "Assignment":
        return self

    def action(
        self,
        current_position: Coord,
        positions: Dict[Union["Resource", Worker], Coord],
        nexus_positions: List[Coord],
    ) -> "Movement":
        target_position = positions[self]
        if current_position == target_position:
            target_position = get_nearest(current_position, nexus_positions)
        return Movement.from_(current_position, to=target_position)

    @property
    def symbol(self):
        if self is Resource.GAS:
            return fg("green") + "G" + RESET
        if self is Resource.MINERALS:
            return fg("blue") + "M" + RESET
        raise RuntimeError

    def __eq__(self, other):
        return Enum.__eq__(self, other)


@dataclass(frozen=True)
class Resources:
    minerals: int
    gas: int

    def as_dict(self) -> Dict[Resource, int]:
        return {Resource.MINERALS: self.minerals, Resource.GAS: self.gas}

    def as_counter(self) -> typing.Counter[Resource]:
        return Counter(self.as_dict())


assert set(Resources(0, 0).__annotations__.keys()) == {
    r.lower() for r in Resource.__members__
}


class Building(WorldObject, Target, WorkerAction, ABC):
    def assignment(self, action3: Optional["IJAction"]) -> "Assignment":
        assert isinstance(action3, IJAction)
        return BuildOrder(building=self, location=(action3.i, action3.j))

    @property
    @abstractmethod
    def cost(self) -> Resources:
        pass

    @property
    @abstractmethod
    def symbol(self) -> str:
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type)

    def __str__(self):
        return self.__class__.__name__


class Assimilator(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=1, gas=0)

    @property
    def symbol(self) -> str:
        return "a"


class CyberneticsCore(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=0)

    @property
    def symbol(self) -> str:
        return "C"


class DarkShrine(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=2)

    @property
    def symbol(self) -> str:
        return "D"


class Forge(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=0)

    @property
    def symbol(self) -> str:
        return "f"


class FleetBeacon(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=3, gas=2)

    @property
    def symbol(self) -> str:
        return "b"


class Gateway(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=0)

    @property
    def symbol(self) -> str:
        return "g"


class Nexus(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=4, gas=0)

    @property
    def symbol(self) -> str:
        return "n"


class PhotonCannon(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=0)

    @property
    def symbol(self) -> str:
        return "c"


class Pylon(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=1, gas=0)

    @property
    def symbol(self) -> str:
        return "p"


class RoboticsBay(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=2)

    @property
    def symbol(self) -> str:
        return "B"


class RoboticsFacility(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=1)

    @property
    def symbol(self) -> str:
        return "F"


class StarGate(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=2)

    @property
    def symbol(self) -> str:
        return "S"


class TemplarArchives(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=2)

    @property
    def symbol(self) -> str:
        return "A"


class TwilightCouncil(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=2, gas=1)

    @property
    def symbol(self) -> str:
        return "T"


Buildings: List[Building] = [
    Assimilator(),
    CyberneticsCore(),
    # DarkShrine(),
    FleetBeacon(),
    # Forge(),
    # Gateway(),
    Nexus(),
    PhotonCannon(),
    Pylon(),
    RoboticsBay(),
    RoboticsFacility(),
    StarGate(),
    # TemplarArchives(),
    TwilightCouncil(),
]

Targets = [*Resource, *Buildings]


def get_nearest(current_position: Coord, candidate_positions: List[Coord]) -> Coord:
    nearest = np.argmin(
        np.max(
            np.abs(
                np.expand_dims(np.array(current_position), 0)
                - np.stack(candidate_positions),
            ),
            axis=-1,
        )
    )
    return candidate_positions[int(nearest)]


@dataclass(frozen=True)
class BuildOrder(Assignment):
    building: Building
    location: Tuple[int, int] = None

    def action(self, current_position: Coord, *args, **kwargs) -> "WorkerAction":
        if current_position == self.location:
            return self.building
        return Movement.from_(current_position, to=self.location)


Assignment = Union[BuildOrder, Resource]


class MovementType(type):
    def __iter__(self):
        for i in range(-1, 2):
            for j in range(-1, 2):
                yield Movement(i, j)


@dataclass(eq=True, frozen=True)
class Movement(WorkerAction, metaclass=MovementType):
    x: int
    y: int

    @classmethod
    def from_(cls, origin, to):
        return cls(*np.clip(np.array(to) - np.array(origin), -1, 1))


WorkerActions = [*Buildings, *Movement]

O = typing.TypeVar("O", Space, torch.Tensor, np.ndarray)


@dataclass(frozen=True)
class Obs(typing.Generic[O]):
    action_mask: O
    can_open_gate: O
    line_mask: O
    lines: O
    next_actions: O
    obs: O
    partial_action: O
    ptr: O
    resources: O


X = typing.TypeVar("X")


class ActionType(type):
    pass


@dataclass(frozen=True)
class Action(metaclass=ActionType):
    @classmethod
    def parse(cls, a) -> "Action":
        assert 0 <= a < cls.size_a()
        # noinspection PyArgumentList
        return cls(*np.unravel_index(int(a), astuple(cls.num_values())))

    @classmethod
    def size_a(cls) -> int:
        return int(np.prod(astuple(cls.num_values())))

    @classmethod
    def mask(cls, size):
        for i in range(size):
            yield i >= cls.size_a()

    @classmethod
    def can_open_gate(cls, size) -> Generator[bool, None, None]:
        for i in range(size):
            try:
                yield cls.parse(i).reset()
            except AssertionError:
                yield False

    @classmethod
    @abstractmethod
    def num_values(cls) -> "Action":
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_ints(self) -> Generator[int, None, None]:
        yield from map(int, astuple(self))

    def next(self) -> ActionType:
        if self.reset():
            return next(CompoundAction.classes())
        return self.next_if_not_reset()

    @abstractmethod
    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError


@dataclass(frozen=True)
class IsOpAction(Action):
    is_op: X

    def to_ints(self) -> Generator[int, None, None]:
        yield int(self.is_op)

    @classmethod
    def num_values(cls) -> "IsOpAction":
        return cls(2)

    def reset(self):
        return not self.is_op

    def next_if_not_reset(self) -> ActionType:
        return WorkerTargetAction


@dataclass(frozen=True)
class WorkerTargetAction(Action):
    worker: X
    target: X

    @classmethod
    def parse(cls, a) -> "Action":
        ints = super().parse(a)
        assert isinstance(ints, cls)
        return cls(worker=Worker(ints.worker + 1), target=Targets[ints.target])

    @classmethod
    def num_values(cls) -> "WorkerTargetAction":
        return cls(worker=len(Worker), target=len(Targets))

    def reset(self):
        return isinstance(self.target, Resource)

    def next_if_not_reset(self) -> ActionType:
        return IJAction

    def to_ints(self) -> Generator[int, None, None]:
        if isinstance(self.worker, Worker):
            yield self.worker.value
        else:
            yield int(self.worker)
        if isinstance(self.target, Target):
            yield Targets.index(self.target)
        else:
            yield int(self.target)


@dataclass(frozen=True)
class IJAction(Action):
    i: X
    j: X

    def to_ints(self) -> Generator[int, None, None]:
        yield self.i
        yield self.j

    @classmethod
    def num_values(cls) -> "IJAction":
        return cls(i=WORLD_SIZE, j=WORLD_SIZE)

    def reset(self):
        return True

    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError


@dataclass(frozen=True)
class RecurringActions(typing.Generic[X]):
    delta: X
    dg: X
    ptr: X


@dataclass(frozen=True)
class RawAction(RecurringActions):
    a: X


@dataclass(frozen=True)
class CompoundAction:
    action1: WorkerTargetAction = None
    action2: IJAction = None
    ptr: int = 0
    active: ActionType = WorkerTargetAction

    @classmethod
    def classes(cls):
        for f in fields(cls):
            if issubclass(f.type, Action):
                yield f.type

    def actions(self):
        for f in fields(self):
            if issubclass(f.type, Action):
                yield getattr(self, f.name)

    def partial_actions(self):
        index = [*self.classes()].index(self.active)
        actions = [*self.actions()][:index]
        for cls, action in itertools.zip_longest(self.classes(), actions):
            if isinstance(action, Action):
                yield from action.to_ints()
            elif issubclass(cls, Action):
                assert action is None
                yield from [0 for _ in astuple(cls.num_values())]
            else:
                raise RuntimeError

    def update(self, action: Union[RawAction, "CompoundAction"]):
        if isinstance(action, CompoundAction):
            return action
        ptr = action.ptr
        assert issubclass(self.active, Action)
        action = self.active.parse(action.a)
        index = [*self.classes()].index(self.active)
        new_keys = (f.name for f in fields(self))
        new_actions = [*[*self.actions()][:index], action]
        assert None not in new_actions
        kwargs = dict(itertools.zip_longest(new_keys, new_actions))
        kwargs.update(active=action.next(), ptr=ptr)
        # noinspection PyTypeChecker
        return replace(self, **kwargs)

    def can_open_gate(self, size):
        assert issubclass(self.active, Action)
        return self.active.can_open_gate(size)

    def mask(self, size):
        assert issubclass(self.active, Action)
        return self.active.mask(size)

    def is_op(self):
        return self.active is next(self.classes())

    def worker(self) -> Worker:
        assert self.action1.worker is not None
        return self.action1.worker

    def assignment(self) -> Assignment:
        assert isinstance(self.action1.target, Target)
        return self.action1.target.assignment(self.action2)


@dataclass(frozen=True)
class Command:
    worker: Worker
    assignment: Assignment


# Check that fields are alphabetical. Necessary because of the way
# that observation gets vectorized.
annotations = Obs.__annotations__
assert tuple(annotations) == tuple(sorted(annotations))


@dataclass(frozen=True)
class Line:
    required: bool
    building: Building


@dataclass
class State:
    action: CompoundAction
    building_positions: Dict[Coord, Building]
    next_action: Dict[Worker, WorkerAction]
    pointer: int
    positions: Dict[Union[Resource, Worker], Coord]
    resources: typing.Counter[Resource]
    success: bool
    time_remaining: int


WorldObjects = list(Buildings) + list(Resource) + list(Worker)


@dataclass
class RecurrentState(Generic[X]):
    a: X
    d: X
    h: X
    dg: X
    p: X
    v: X
    a_probs: X
    d_probs: X
    dg_probs: X


@dataclass
class ParsedInput(Generic[X]):
    obs: X
    actions: X


@dataclass
class CurriculumSetting:
    max_build_tree_depth: int
    max_lines: int
    n_lines_space: Discrete
    level: int

    def increment_max_lines(self) -> "CurriculumSetting":
        low = self.n_lines_space.low
        high = min(self.n_lines_space.high + 1, self.max_lines)
        n_lines_space = Discrete(low=low, high=high)
        return replace(self, n_lines_space=n_lines_space)

    def increment_build_tree_depth(self) -> "CurriculumSetting":
        return replace(self, max_build_tree_depth=self.max_build_tree_depth + 1)

    def increment_level(self) -> "CurriculumSetting":
        return replace(self, level=self.level + 1)
