import itertools
import typing
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, astuple, fields
from enum import unique, Enum, auto, EnumMeta
from typing import Tuple, Union, List, Generator, Dict, Generic

import numpy as np
import torch
from colored import fg
from gym import Space

from utils import RESET

Coord = Tuple[int, int]


@unique
class Unit(Enum):
    WORKER = auto()


@unique
class WorkerID(Enum):
    A = auto()
    B = auto()
    C = auto()


class Assignment:
    @abstractmethod
    def action(
        self,
        current_position: Coord,
        positions: Dict[Union["Resource", WorkerID], Coord],
        nexus_positions: List[Coord],
    ) -> "WorkerAction":
        raise NotImplementedError


class Target:
    @abstractmethod
    def assignment(self, coord: Coord) -> "Assignment":
        raise NotImplementedError

    @classmethod
    def index(cls, item: "Target") -> int:
        return [*cls].index(item)


class WorkerAction:
    pass


@unique
class Building(Target, WorkerAction, Enum):
    PYLON = auto()
    ASSIMILATOR = auto()
    NEXUS = auto()
    FORGE = auto()
    PHOTON_CANNON = auto()
    GATEWAY = auto()
    CYBERNETICS_CORE = auto()
    TWILIGHT_COUNCIL = auto()
    TEMPLAR_ARCHIVES = auto()
    DARK_SHRINE = auto()
    STARGATE = auto()
    FLEET_BEACON = auto()
    ROBOTICS_FACILITY = auto()
    ROBOTICS_BAY = auto()

    def assignment(self, coord: Coord) -> "Assignment":
        return BuildOrder(building=self, location=coord)


@unique
class Resource(Target, Assignment, Enum):
    MINERALS = auto()
    GAS = auto()

    def assignment(self, coord: Coord) -> "Assignment":
        return self

    def action(
        self,
        current_position: Coord,
        positions: Dict[Union["Resource", WorkerID], Coord],
        nexus_positions: List[Coord],
    ) -> "Movement":
        target_position = positions[self]
        if current_position == target_position:
            nearest = int(
                np.argmin(
                    np.max(
                        np.abs(
                            np.expand_dims(np.array(current_position), 0)
                            - np.stack(nexus_positions),
                        ),
                        axis=-1,
                    )
                )
            )
            target_position = nexus_positions[nearest]
        return Movement.from_(current_position, to=target_position)


Targets = [*Resource, *Building]


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


WorkerActions = [*Building, *Movement]

O = typing.TypeVar("O", Space, torch.Tensor, np.ndarray)


@dataclass(frozen=True)
class Obs(typing.Generic[O]):
    action_mask: O
    can_open_gate: O
    lines: O
    mask: O
    obs: O
    partial_action: O
    resources: O
    workers: O


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
            yield i < cls.size_a()

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

    def next(self) -> ActionType:
        if self.reset():
            return Action1
        return self.next_if_not_reset()

    @abstractmethod
    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError


@dataclass(frozen=True)
class Action1(Action):
    is_op: X

    @classmethod
    def num_values(cls) -> "Action1":
        return cls(2)

    def reset(self):
        return not self.is_op

    def next_if_not_reset(self) -> ActionType:
        return Action2


@dataclass(frozen=True)
class Action2(Action):
    worker: X
    target: X

    @classmethod
    def parse(cls, a) -> "Action":
        ints = super().parse(a)
        assert isinstance(ints, cls)
        return cls(worker=WorkerID(ints.worker + 1), target=Targets[ints.target])

    @classmethod
    def num_values(cls) -> "Action2":
        return cls(worker=len(WorkerID), target=len(Targets))

    def reset(self):
        return isinstance(Targets[self.target], Resource)

    def next_if_not_reset(self) -> ActionType:
        return Action3


WORLD_SIZE = 0


@dataclass(frozen=True)
class Action3(Action):
    i: X
    j: X

    @classmethod
    def num_values(cls) -> "Action3":
        return cls(i=WORLD_SIZE, j=WORLD_SIZE)

    def reset(self):
        return True

    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError


@dataclass(frozen=True)
class RecurringActions(typing.Generic[X]):
    delta: X
    ptr: X


@dataclass(frozen=True)
class RawAction(RecurringActions):
    a: X


@dataclass(frozen=True)
class CompoundAction:
    action1: Action1 = None
    action2: Action2 = None
    action3: Action3 = None
    ptr: int = 0
    active: ActionType = Action1

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
                yield from [1 + x for x in astuple(action)]
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
        filled_in = [*self.actions()][:index]
        assert None not in filled_in
        # noinspection PyTypeChecker
        return CompoundAction(*filled_in, action, active=action.next(), ptr=ptr)

    def can_open_gate(self, size):
        assert issubclass(self.active, Action)
        return self.active.can_open_gate(size)

    def mask(self, size):
        assert issubclass(self.active, Action)
        return self.active.mask(size)

    def is_op(self):
        return None not in astuple(self)

    def worker(self) -> WorkerID:
        assert self.action2.worker is not None
        return self.action2.worker

    def assignment(self) -> Assignment:
        assert isinstance(self.action2.target, Target)
        return self.action2.target.assignment(astuple(self.action3))


@dataclass(frozen=True)
class Command:
    worker: WorkerID
    assignment: Assignment


@dataclass(frozen=True)
class Resources:
    minerals: int
    gas: int

    def as_dict(self) -> Dict[Resource, int]:
        return {Resource.MINERALS: self.minerals, Resource.GAS: self.gas}


assert set(Resources(0, 0).__annotations__.keys()) == {
    r.lower() for r in Resource.__members__
}


# Check that fields are alphabetical. Necessary because of the way
# that observation gets vectorized.
annotations = Obs.__annotations__
assert tuple(annotations) == tuple(sorted(annotations))

costs = {
    Building.NEXUS: Resources(minerals=4, gas=0),
    Building.PYLON: Resources(minerals=1, gas=0),
    Building.ASSIMILATOR: Resources(minerals=1, gas=0),
    Building.FORGE: Resources(minerals=2, gas=0),
    Building.GATEWAY: Resources(minerals=2, gas=0),
    Building.CYBERNETICS_CORE: Resources(minerals=2, gas=0),
    Building.PHOTON_CANNON: Resources(minerals=2, gas=0),
    Building.TWILIGHT_COUNCIL: Resources(minerals=2, gas=1),
    Building.STARGATE: Resources(minerals=2, gas=2),
    Building.ROBOTICS_FACILITY: Resources(minerals=2, gas=1),
    Building.TEMPLAR_ARCHIVES: Resources(minerals=2, gas=2),
    Building.DARK_SHRINE: Resources(minerals=2, gas=2),
    Building.ROBOTICS_BAY: Resources(minerals=2, gas=2),
    Building.FLEET_BEACON: Resources(minerals=3, gas=2),
}

Costs: Dict[Building, typing.Counter[Resource]] = {
    b: Counter(c.as_dict()) for b, c in costs.items()
}

# ensure all buildings are in costs
assert len(Costs) == len(Building)


@dataclass(frozen=True)
class Line:
    required: bool
    building: Building


@dataclass(frozen=True)
class Leaf:
    building: Building


@dataclass(frozen=True)
class Node:
    building: Building
    children: "Tree"


Tree = List[Union[Node, Leaf]]

build_tree = [
    Leaf(Building.PYLON),
    Leaf(Building.ASSIMILATOR),
    Node(
        Building.NEXUS,
        [
            Node(Building.FORGE, [Leaf(Building.PHOTON_CANNON)]),
            Node(
                Building.GATEWAY,
                [
                    Node(
                        Building.CYBERNETICS_CORE,
                        [
                            Node(
                                Building.TWILIGHT_COUNCIL,
                                [
                                    Leaf(Building.TEMPLAR_ARCHIVES),
                                    Leaf(Building.DARK_SHRINE),
                                ],
                            ),
                            Node(Building.STARGATE, [Leaf(Building.FLEET_BEACON)]),
                            Node(
                                Building.ROBOTICS_FACILITY,
                                [Leaf(Building.ROBOTICS_BAY)],
                            ),
                        ],
                    )
                ],
            ),
        ],
    ),
]


def flatten(tree: Tree) -> Generator[Building, None, None]:
    for node in tree:
        yield node.building
        if isinstance(node, Node):
            yield from flatten(node.children)


# ensure all buildings are in build_tree
assert set(flatten(build_tree)) == set(Building)


@dataclass
class State:
    action: CompoundAction
    building_positions: Dict[Coord, Building]
    next_action: Dict[WorkerID, WorkerAction]
    pointer: int
    positions: Dict[Union[Resource, WorkerID], Coord]
    resources: typing.Counter[Resource]
    success: bool


WorldObject = Union[Building, Resource, WorkerID]
WorldObjects = list(Building) + list(Resource) + list(WorkerID)

Symbols: Dict[WorldObject, Union[str, int]] = {
    Building.PYLON: "p",
    Building.ASSIMILATOR: "a",
    Building.NEXUS: "n",
    Building.FORGE: "f",
    Building.PHOTON_CANNON: "c",
    Building.GATEWAY: "g",
    Building.CYBERNETICS_CORE: "C",
    Building.TWILIGHT_COUNCIL: "T",
    Building.TEMPLAR_ARCHIVES: "A",
    Building.DARK_SHRINE: "D",
    Building.STARGATE: "S",
    Building.FLEET_BEACON: "b",
    Building.ROBOTICS_FACILITY: "F",
    Building.ROBOTICS_BAY: "B",
    WorkerID.A: 1,
    WorkerID.B: 1,
    WorkerID.C: 1,
    Resource.GAS: fg("green") + "G" + RESET,
    Resource.MINERALS: fg("blue") + "M" + RESET,
}

assert set(Symbols) == set(WorldObjects)


@dataclass
class RecurrentState(Generic[X]):
    a: X
    d: X
    h: X
    dg: X
    p: X
    v: X
    l: X
    a_probs: X
    d_probs: X
    dg_probs: X


@dataclass
class ParsedInput(Generic[X]):
    obs: X
    actions: X
