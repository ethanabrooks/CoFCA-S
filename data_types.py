import typing
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, asdict, astuple, replace, fields
from enum import unique, Enum, auto
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
class Building(Enum):
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


@dataclass(eq=True, frozen=True)
class Movement:
    x: int
    y: int


@unique
class Resource(Enum):
    MINERALS = auto()
    GAS = auto()


@unique
class WorkerID(Enum):
    A = auto()
    B = auto()
    C = auto()


O = typing.TypeVar("O", Space, torch.Tensor, np.ndarray)


@dataclass(frozen=True)
class Obs(typing.Generic[O]):
    lines: O
    mask: O
    obs: O
    resources: O
    workers: O


X = typing.TypeVar("X", np.ndarray, typing.Optional[int], torch.Tensor)

ActionTargets = list(Resource) + list(Building)


@dataclass(frozen=True)
class PartialAction(typing.Generic[X]):
    @staticmethod
    def get_gate_value(x: torch.Tensor) -> torch.Tensor:
        return x % 2

    @classmethod
    def parse(cls, a):
        if cls.can_reset():
            a = a // 2
        assert 0 <= a < cls.size_a()
        # noinspection PyArgumentList
        return cls(*np.unravel_index(int(a), astuple(cls.num_values())))

    @classmethod
    def size_a(cls):
        return np.prod(astuple(cls.num_values())) * (2 ** cls.can_reset())

    @classmethod
    def mask(cls, size):
        mask = np.zeros(size)
        mask[np.arange(size) < cls.size_a()] = 1
        return mask

    @classmethod
    @abstractmethod
    def num_values(cls) -> "PartialAction":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_reset(cls) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> bool:
        raise NotImplementedError

    def next(self) -> type:
        if self.reset():
            assert self.can_reset()
            return Action1
        return self.next_if_not_reset()

    @abstractmethod
    def next_if_not_reset(self) -> type:
        raise NotImplementedError


@dataclass(frozen=True)
class Action1(PartialAction):
    is_op: X

    @classmethod
    def num_values(cls) -> "Action1":
        return cls(2)

    @classmethod
    def can_reset(cls):
        return True

    def reset(self):
        return not self.is_op

    def next_if_not_reset(self) -> type:
        return Action2


@dataclass(frozen=True)
class Action2(PartialAction):
    verb: X

    @classmethod
    def num_values(cls) -> "Action2":
        return cls(3)

    @classmethod
    def can_reset(cls):
        return False

    def reset(self):
        return False

    def next_if_not_reset(self) -> type:
        return Action3


@dataclass(frozen=True)
class Action3(PartialAction):
    noun: X
    # gate: X

    @classmethod
    def num_values(cls) -> "Action3":
        return cls(noun=3)  # , gate=2)

    @classmethod
    def can_reset(cls):
        return True

    def reset(self):
        return True

    def next_if_not_reset(self) -> type:
        raise NotImplementedError


@dataclass(frozen=True)
class RecurringActions(typing.Generic[X]):
    delta: X
    ptr: X


@dataclass(frozen=True)
class RawAction(RecurringActions):
    a: X


@dataclass(frozen=True)
class VariableActions:
    action1: Action1 = None
    action2: Action2 = None
    action3: Action3 = None

    def verb(self):
        return self.action2.verb

    def noun(self):
        return self.action3.noun

    @classmethod
    def classes(cls):
        for f in fields(cls):
            if issubclass(f.type, PartialAction):
                yield f.type

    def actions(self):
        for f in fields(self):
            yield getattr(self, f.name)

    def partial_actions(self):
        for cls, action in zip(self.classes(), self.actions()):
            if isinstance(action, PartialAction):
                yield from [1 + x for x in astuple(action)]
            elif issubclass(cls, PartialAction):
                assert action is None
                yield from [0 for _ in astuple(cls.num_values())]
            else:
                raise RuntimeError

    def update(self, next_action: PartialAction):
        index = [*self.classes()].index(type(next_action))
        filled_in = [*self.actions()][:index]
        assert None not in filled_in
        return VariableActions(*filled_in, next_action)

    def no_op(self):
        return None in astuple(self)


@dataclass(frozen=True)
class NonAAction(typing.Generic[X]):
    delta: X
    dg: X
    ptr: X


@dataclass(frozen=True)
class RawAction(NonAAction):
    a: X


@dataclass(frozen=True)
class AActions(typing.Generic[X]):
    is_op: X  # 2
    verb: X
    noun: X
    # target: X  # 16
    # worker: X  # 3
    # ij: X  # 64

    def targeted(self):
        return ActionTargets[self.target]

    def no_op(self):
        return not self.is_op or None in (self.verb, self.noun)

    def complete(self):
        return self.next_key() is "is_op"

    @staticmethod
    def to_int(x):
        return 0 if x is None else 1 + x

    def to_array(self):
        return np.array([*map(self.to_int, astuple(self))])

    def next_key(self):
        if self.is_op in (None, 0):
            return "is_op"
        if self.verb is None:
            return "verb"
        if self.noun is None:
            return "noun"
        return "is_op"


@dataclass(frozen=True)
class Action(AActions, NonAAction):
    @staticmethod
    def none_action():
        return Action(None, None, None, None, None, None)

    def a_actions(self):
        return AActions(
            **{k: v for k, v in asdict(self).items() if k in AActions.__annotations__}
        )

    def parse(self, world_shape: Coord):
        if not self.is_op or any(x < 0 for x in astuple(self)):
            return None
        action_target = self.targeted()
        if action_target in Building:
            i, j = np.unravel_index(self.ij, world_shape)
            assignment = BuildOrder(building=action_target, location=(i, j))
        elif action_target in Resource:
            assignment = action_target
        else:
            raise RuntimeError
        return Command(WorkerID(self.worker + 1), assignment)


@dataclass(frozen=True)
class BuildOrder:
    building: Building
    location: Tuple[int, int] = None


Assignment = Union[BuildOrder, Resource]


@dataclass
class Worker:
    assignment: Assignment
    next_action: Union[Movement, Building] = None

    def get_action(
        self,
        position: Coord,
        positions: Dict[Union[Resource, WorkerID], Coord],
        nexus_positions: List[Coord],
    ):
        if isinstance(self.assignment, Resource):
            objective = positions[self.assignment]
            if position == objective:
                nearest = int(
                    np.argmin(
                        np.max(
                            np.abs(
                                np.expand_dims(np.array(position), 0)
                                - np.stack(nexus_positions),
                            ),
                            axis=-1,
                        )
                    )
                )
                goto = nexus_positions[nearest]
            else:
                goto = objective
        elif isinstance(self.assignment, BuildOrder):
            goto = self.assignment.location
        else:
            raise RuntimeError
        if goto == position:
            if isinstance(self.assignment, BuildOrder):
                return self.assignment.building
        return Movement(*np.clip(np.array(goto) - np.array(position), -1, 1))


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


def worker_actions():
    yield from Building
    for i in range(-1, 2):
        for j in range(-1, 2):
            yield Movement(i, j)


@dataclass
class State:
    building_positions: Dict[Coord, Building]
    positions: Dict[Union[Resource, WorkerID], Coord]
    resources: typing.Counter[Resource]
    workers: Dict[WorkerID, Worker]
    success: bool
    pointer: int


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
    a_probs: X
    d_probs: X
    dg_probs: X


@dataclass
class ParsedInput(Generic[X]):
    obs: X
    actions: X
