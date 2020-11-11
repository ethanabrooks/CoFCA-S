import collections
from collections import namedtuple, Counter
from dataclasses import dataclass
from enum import unique, Enum, auto
from typing import Tuple, Union, List, Generator, Dict
import numpy as np
import typing

from colored import fg
from ray.rllib.train import torch

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


X = Union[int, torch.Tensor]


@dataclass(frozen=True)
class Action:
    delta: X
    dg: X
    type: X  # 2
    worker: X  # 3
    target: X  # 16
    i: X  # 8
    j: X  # 8

    def parse(self):
        action_type = ActionTypes[self.type]
        if action_type is None:
            return action_type
        elif action_type is Command:
            action_target = ActionTargets[self.target]
            if action_target in Building:
                assignment = BuildOrder(
                    building=action_target, location=(self.i, self.j)
                )
            elif action_target in Resource:
                assignment = action_target
            else:
                raise RuntimeError
            return Command(WorkerID(self.worker + 1), assignment)
        else:
            raise RuntimeError


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


Obs = namedtuple("Obs", "lines mask obs resources workers")

# Check that fields are alphabetical. Necessary because of the way
# that observation gets vectorized.
assert tuple(Obs._fields) == tuple(sorted(Obs._fields))

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
ActionTypes = [None, Command]
ActionTargets = list(Building) + list(Resource)

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
