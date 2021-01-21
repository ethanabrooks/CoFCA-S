import itertools
import typing
from abc import abstractmethod, ABC, ABCMeta
from collections import Counter
from dataclasses import dataclass, astuple, replace, field
from enum import unique, Enum, auto, EnumMeta
from functools import lru_cache
from typing import Tuple, Union, List, Generator, Dict, Generic, Optional

import gym
import numpy as np
import torch
from colored import fg
from gym import spaces

from utils import RESET

CoordType = Tuple[int, int]
IntGenerator = Generator[int, None, None]
IntListGenerator = Generator[List[int], None, None]
BoolGenerator = Generator[bool, None, None]

WORLD_SIZE = None


def move_from(origin: CoordType, toward: CoordType) -> CoordType:
    origin = np.array(origin)
    i, j = np.array(origin) + np.clip(
        np.array(toward) - origin,
        -1,
        1,
    )
    return i, j


class InvalidInput(Exception):
    pass


""" abstract classes """


class WorldObject:
    @property
    @abstractmethod
    def symbol(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class ActionComponentMeta(type):
    pass


class ActionComponentEnumMeta(ActionComponentMeta, EnumMeta):
    pass


class ActionComponentABCMeta(ActionComponentMeta, ABCMeta):
    pass


class ActionComponent(metaclass=ActionComponentMeta):
    @staticmethod
    @abstractmethod
    def parse(n: int) -> "ActionComponent":
        pass

    @staticmethod
    @abstractmethod
    def space() -> spaces.Discrete:
        pass

    @abstractmethod
    def to_int(self) -> int:
        pass


ActionComponentGenerator = Generator[ActionComponent, None, None]


class Building(WorldObject, ActionComponent, ABC, metaclass=ActionComponentABCMeta):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"({Buildings.index(self)}) {str(self)}: {self.cost}"

    @property
    @abstractmethod
    def cost(self) -> "Resources":
        pass

    def on(self, coord: "CoordType", building_positions: "BuildingPositions"):
        return self == building_positions.get(coord)

    @staticmethod
    def parse(n: int) -> "Building":
        return Buildings[n]

    @staticmethod
    def space() -> spaces.Discrete:
        return spaces.Discrete(len(Buildings))

    @property
    @abstractmethod
    def symbol(self) -> str:
        pass

    def to_int(self) -> int:
        return Buildings.index(self)


class Assignment:
    def execute(
        self,
        assignments: "Assignments",
        resources: typing.Counter["Resource"],
        worker: "Worker",
        **kwargs,
    ) -> Optional[str]:
        original_assignment = assignments[worker]
        error_msg = self._execute(
            assignments=assignments, resources=resources, worker=worker, **kwargs
        )
        if error_msg is None and isinstance(original_assignment, BuildOrder):
            # refund cost of building since assignment changed.
            resources.update(original_assignment.building.cost)
        return error_msg

    @abstractmethod
    def _execute(
        self,
        positions: "Positions",
        worker: "Worker",
        assignments: "Assignments",
        building_positions: "BuildingPositions",
        pending_positions: "BuildingPositions",
        required: typing.Counter["Building"],
        resources: typing.Counter["Resource"],
        carrying: "Carrying",
    ) -> Optional[str]:
        raise NotImplementedError


""" world objects"""


@unique
class Worker(WorldObject, ActionComponent, Enum, metaclass=ActionComponentEnumMeta):
    W1 = auto()
    W2 = auto()
    W3 = auto()

    # W4 = auto()
    # W5 = auto()
    # W6 = auto()
    # W7 = auto()
    # W8 = auto()
    # W9 = auto()
    # W10 = auto()
    # W11 = auto()
    # W12 = auto()

    def __eq__(self, other):
        # noinspection PyArgumentList
        return Enum.__eq__(self, other)

    def __lt__(self, other):
        assert isinstance(other, Worker)
        # noinspection PyArgumentList
        return self.value < other.value

    def __hash__(self):
        # noinspection PyArgumentList
        return Enum.__hash__(self)

    def on(
        self,
        coord: "CoordType",
        positions: "Positions",
    ) -> bool:
        return positions[self] == coord

    @staticmethod
    def parse(n: int) -> "Worker":
        return Worker(n)

    @staticmethod
    def space() -> spaces.Discrete:
        return spaces.Discrete(len(Worker))  # binary: in or out

    @property
    def symbol(self) -> str:
        return str(self.value)

    def to_int(self) -> int:
        return self.value


WorkerGenerator = Generator[Worker, None, None]


@unique
class Resource(WorldObject, Assignment, Enum):
    MINERALS = auto()
    GAS = auto()

    def __hash__(self):
        return Enum.__hash__(self)

    def __eq__(self, other):
        return Enum.__eq__(self, other)

    def _execute(
        self,
        positions: "Positions",
        worker: "Worker",
        assignments: "Assignments",
        building_positions: "BuildingPositions",
        pending_positions: "BuildingPositions",
        required: typing.Counter["Building"],
        resources: typing.Counter["Resource"],
        carrying: "Carrying",
    ) -> Optional[str]:
        worker_pos = positions[worker]

        if carrying[worker] is None:
            resource_pos = positions[self]
            positions[worker] = move_from(worker_pos, toward=resource_pos)
            worker_pos = positions[worker]
            if worker_pos == resource_pos:
                if self is Resource.GAS and not isinstance(
                    building_positions.get(positions[worker]), Assimilator
                ):
                    return "Assimilator required for harvesting gas"  # no op on gas unless Assimilator
                carrying[worker] = self
        else:
            nexus_positions: List[CoordType] = [
                p for p, b in building_positions.items() if isinstance(b, Nexus)
            ]
            nexus = get_nearest(nexus_positions, to=worker_pos)
            positions[worker] = move_from(
                worker_pos,
                toward=nexus,
            )
            if positions[worker] == nexus:
                resource = carrying[worker]
                assert isinstance(resource, Resource)
                resources[resource] += 100
                carrying[worker] = None
        return None

    def on(
        self,
        coord: "CoordType",
        positions: "Positions",
    ) -> bool:
        return positions[self] == coord

    @property
    def symbol(self) -> str:
        if self is Resource.GAS:
            return fg("green") + "g" + RESET
        if self is Resource.MINERALS:
            return fg("blue") + "m" + RESET
        raise RuntimeError


ResourceCounter = typing.Counter[Resource]


@dataclass(frozen=True)
class Resources:
    minerals: int
    gas: int

    def __iter__(self):
        yield from [Resource.MINERALS] * self.minerals
        yield from [Resource.GAS] * self.gas


assert set(Resources(0, 0).__annotations__.keys()) == {
    r.lower() for r in Resource.__members__
}

""" action components """


@dataclass
class Coord(ActionComponent):
    i: int
    j: int

    @staticmethod
    def parse(n: int) -> "Coord":
        assert isinstance(WORLD_SIZE, int)
        ij = np.unravel_index(n, (WORLD_SIZE, WORLD_SIZE))
        return Coord(*ij)

    @staticmethod
    def possible_values():
        assert isinstance(WORLD_SIZE, int)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                yield i, j

    @staticmethod
    def space() -> spaces.Discrete:
        assert isinstance(WORLD_SIZE, int)
        return spaces.Discrete(WORLD_SIZE ** 2)

    def to_int(self) -> int:
        return int(np.ravel_multi_index((self.i, self.j), (WORLD_SIZE, WORLD_SIZE)))

    @staticmethod
    def zeros() -> IntGenerator:
        yield 0
        yield 0


BuildingPositions = Dict[CoordType, Building]
Positions = Dict[Union[Resource, Worker], CoordType]
Carrying = Dict[Worker, Optional[Resource]]
Assignments = Dict[Worker, Assignment]


@dataclass(frozen=True)
class BuildOrder(Assignment):
    building: Building
    coord: CoordType

    def _execute(
        self,
        positions: "Positions",
        worker: "Worker",
        assignments: "Assignments",
        building_positions: "BuildingPositions",
        pending_positions: "BuildingPositions",
        required: typing.Counter["Building"],
        resources: typing.Counter["Resource"],
        carrying: "Carrying",
    ) -> Optional[str]:
        if self.coord not in pending_positions:
            pending_positions[self.coord] = self.building
            resources.subtract(self.building.cost)
        if positions[worker] == self.coord:
            building_positions[self.coord] = self.building
            del pending_positions[self.coord]
            assignments[worker] = DoNothing()
            return None
        else:
            return GoTo(self.coord).execute(
                positions=positions,
                worker=worker,
                assignments=assignments,
                building_positions=building_positions,
                pending_positions=pending_positions,
                required=required,
                resources=resources,
                carrying=carrying,
            )


@dataclass(frozen=True)
class GoTo(Assignment):
    coord: CoordType

    def _execute(
        self, positions: "Positions", worker: "Worker", assignments, *args, **kwargs
    ) -> Optional[str]:
        positions[worker] = move_from(positions[worker], toward=self.coord)
        return


class DoNothing(Assignment):
    def _execute(self, *args, **kwargs) -> Optional[str]:
        return


Command = Union[BuildOrder, Resource]

O = typing.TypeVar("O", torch.Tensor, np.ndarray, int, gym.Space)


@dataclass(frozen=True)
class Obs(typing.Generic[O]):
    action_mask: O
    gate_openers: O
    line_mask: O
    lines: O
    obs: O
    partial_action: O
    ptr: O
    resources: O


X = typing.TypeVar("X")


@dataclass(frozen=True)
class RawAction:
    delta: Union[np.ndarray, torch.Tensor, X]
    dg: Union[np.ndarray, torch.Tensor, X]
    ptr: Union[np.ndarray, torch.Tensor, X]
    a: Union[np.ndarray, torch.Tensor, X]

    @staticmethod
    def parse(*xs) -> "RawAction":
        delta, dg, ptr, *a = xs
        if a == [None]:
            a = None
        return RawAction(delta, dg, ptr, a)

    def flatten(self) -> Generator[any, None, None]:
        yield from astuple(self)


Ob = Optional[bool]
OB = Optional[Building]
OC = Optional[Coord]


@dataclass(frozen=True)
class CompoundAction:
    worker_values: List[Ob] = field(default_factory=lambda: [False for _ in Worker])
    building: OB = None
    coord: OC = None

    @staticmethod
    def _worker_values() -> List[Ob]:
        return [False, True]

    @classmethod
    def input_space(cls):
        return spaces.MultiDiscrete(
            [
                *[len(cls._worker_values()) for _ in Worker],
                1 + Coord.space().n + Building.space().n,
            ]
        )

    @classmethod
    def parse(cls, *values: int):
        *worker_values, building_or_coord = values
        worker_values = [cls._worker_values()[w] for w in worker_values]
        if building_or_coord == 0:
            coord = building = None
        else:
            building_or_coord -= 1
            if Coord.space().contains(building_or_coord):
                coord = Coord.parse(building_or_coord)
                building = None
            else:
                building_or_coord -= Coord.space().n
                if Building.space().contains(building_or_coord):
                    coord = None
                    building = Building.parse(building_or_coord)
                else:
                    raise RuntimeError

        return CompoundAction(
            worker_values=worker_values, coord=coord, building=building
        )

    @classmethod
    def representation_space(cls):
        assert isinstance(WORLD_SIZE, int)
        return spaces.MultiDiscrete(
            [
                *[len(cls._worker_values()) for _ in Worker],
                1 + WORLD_SIZE,
                1 + WORLD_SIZE,
                1 + len(Buildings),
            ]
        )

    def to_input_int(self) -> IntGenerator:
        for w in self.worker_values:
            yield self._worker_values().index(w)
        if (self.coord, self.building) == (None, None):
            yield 0
        elif self.building is None:
            yield 1 + self.coord.to_int()
        elif self.coord is None:
            yield 1 + Coord.space().n + self.building.to_int()
        else:
            raise NotImplementedError("self.coord or self.building must be None.")

    def to_representation_ints(self) -> IntGenerator:
        for w in self.worker_values:
            yield self._worker_values().index(w)
        yield from (
            (0, 0) if self.coord is None else (1 + c for c in astuple(self.coord))
        )
        yield 0 if self.building is None else 1 + self.building.to_int()


CompoundActionGenerator = Generator[CompoundAction, None, None]


@dataclass(frozen=True)
class ActionStage:
    @staticmethod
    def _children() -> List[type]:
        return [
            NoWorkersAction,
            WorkersAction,
            BuildingAction,
            CoordAction,
            BuildingCoordAction,
        ]

    @staticmethod
    @abstractmethod
    def _gate_openers() -> CompoundActionGenerator:
        pass

    @staticmethod
    @abstractmethod
    def _parse_string(s: str) -> CompoundAction:
        pass

    @staticmethod
    @abstractmethod
    def _permitted_values() -> CompoundActionGenerator:
        pass

    @staticmethod
    @abstractmethod
    def _prompt() -> str:
        pass

    @staticmethod
    @abstractmethod
    def _update(action: CompoundAction) -> "ActionStage":
        pass

    @abstractmethod
    def action_components(self) -> CompoundAction:
        pass

    @abstractmethod
    def assignment(self, positions: Positions) -> Optional[Assignment]:
        pass

    def from_input(self) -> "ActionStage":
        compound_action = None
        while compound_action is None:
            string = input(self._prompt() + "\n")
            if string:
                try:
                    compound_action = self._parse_string(string)
                except InvalidInput as e:
                    print(e)
            else:
                compound_action = CompoundAction()
        return self._update(compound_action)

    @classmethod
    @lru_cache
    def gate_openers(cls) -> np.ndarray:
        return np.array([list(o.to_input_int()) for o in cls._gate_openers()])

    @classmethod
    def gate_opener_max_size(cls):
        def opener_size():
            for c in cls._children():
                assert issubclass(c, ActionStage)
                yield len(c.gate_openers())

        return max(opener_size())

    @abstractmethod
    def get_workers(self) -> WorkerGenerator:
        raise NotImplementedError

    def invalid(
        self,
        resources: typing.Counter[Resource],
        dependencies: Dict[Building, Building],
        building_positions: BuildingPositions,
        pending_positions: BuildingPositions,
        positions: Positions,
    ) -> Optional[str]:
        return

    @classmethod
    @lru_cache
    def mask(cls) -> np.ndarray:
        nvec = CompoundAction.input_space().nvec
        mask = np.ones((len(nvec), max(nvec)))
        R = np.arange(len(nvec))
        for permitted_values in cls._permitted_values():
            unmask = [*permitted_values.to_input_int()]
            mask[R, unmask] = 0
        return mask

    def to_ints(self):
        return self.action_components().to_representation_ints()

    def update(self, *components: int) -> "ActionStage":
        return self._update(CompoundAction.parse(*components))


class CoordCanOpenGate(ActionStage, ABC):
    @staticmethod
    def _gate_openers() -> CompoundActionGenerator:
        for i, j in Coord.possible_values():
            yield CompoundAction(coord=Coord(i, j))


@dataclass(frozen=True)
class NoWorkersAction(ActionStage):
    @staticmethod
    def _gate_openers() -> CompoundActionGenerator:
        # selecting no workers is a no-op that allows gate to open
        yield CompoundAction(worker_values=[False for _ in Worker])

    @staticmethod
    def _parse_string(s: str) -> CompoundAction:
        try:
            worker_idxs = {int(c) for c in s.split()}
        except ValueError:
            raise InvalidInput
        workers = [w.value in worker_idxs for w in Worker]
        return CompoundAction(worker_values=workers)

    @staticmethod
    def _permitted_values() -> CompoundActionGenerator:
        for worker_values in itertools.product([True, False], repeat=len(Worker)):
            yield CompoundAction(worker_values=[*worker_values])

    @staticmethod
    def _prompt() -> str:
        return "Workers:"

    def _update(
        self, action: CompoundAction
    ) -> Union["WorkersAction", "NoWorkersAction"]:
        workers = [w for b, w in zip(action.worker_values, Worker) if b]
        if not workers:
            return NoWorkersAction()
        return WorkersAction(workers)

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        return DoNothing()

    def get_workers(self) -> WorkerGenerator:
        yield from ()

    def action_components(self) -> CompoundAction:
        return CompoundAction()


@dataclass(frozen=True)
class HasWorkers(ActionStage, ABC):
    workers: List[Worker]

    def action_components(self) -> CompoundAction:
        return CompoundAction(worker_values=[w in self.workers for w in Worker])

    def get_workers(self) -> WorkerGenerator:
        yield from self.workers


@dataclass(frozen=True)
class WorkersAction(HasWorkers, CoordCanOpenGate):
    @staticmethod
    def _parse_string(s: str) -> CompoundAction:
        try:
            i, j = map(int, s.split())
        except ValueError:
            try:
                n = int(s)
            except ValueError:
                raise InvalidInput
            try:
                building = Buildings[n]
            except IndexError as e:
                raise InvalidInput(e)
            return CompoundAction(building=building)
        return CompoundAction(coord=Coord(i, j))

    @staticmethod
    def _permitted_values() -> CompoundActionGenerator:
        yield CompoundAction()
        for i, j in Coord.possible_values():
            yield CompoundAction(coord=Coord(i, j))
        for building in Buildings:
            yield CompoundAction(building=building)

    @staticmethod
    def _prompt() -> str:
        return "\n".join(
            [f"({i}) {b}" for i, b in enumerate(Buildings)] + ["Coord or Building"]
        )

    def _update(self, action: CompoundAction) -> "ActionStage":
        if (action.coord, action.building) == (None, None):
            assert not any(action.worker_values)
            return NoWorkersAction()
        if action.coord is not None:
            assert not any([*action.worker_values, action.building])
            return CoordAction(workers=self.workers, coord=action.coord)
        if action.building is not None:
            assert not any([*action.worker_values, action.coord])
            return BuildingAction(workers=self.workers, building=action.building)
        raise RuntimeError

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        return None


@dataclass(frozen=True)
class CoordAction(HasWorkers, NoWorkersAction):
    coord: Coord

    def action_components(self) -> CompoundAction:
        return replace(HasWorkers.action_components(self), coord=self.coord)

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        i, j = astuple(self.coord)
        for resource in Resource:
            if resource.on((i, j), positions):
                return resource
        return GoTo((i, j))


@dataclass(frozen=True)
class BuildingAction(HasWorkers, CoordCanOpenGate):
    building: Building

    @staticmethod
    def _parse_string(s: str) -> CompoundAction:
        try:
            i, j = map(int, s.split())
        except ValueError:
            raise InvalidInput
        return CompoundAction(coord=Coord(i, j))

    @staticmethod
    def _permitted_values() -> CompoundActionGenerator:
        yield CompoundAction()
        for i, j in Coord.possible_values():
            yield CompoundAction(coord=Coord(i, j))

    @staticmethod
    def _prompt() -> str:
        return "Coord"

    def _update(self, action: CompoundAction) -> "ActionStage":
        assert not any([*action.worker_values, action.building])
        if action.coord is None:
            return NoWorkersAction()
        return BuildingCoordAction(
            workers=self.workers, building=self.building, coord=action.coord
        )

    def action_components(self) -> CompoundAction:
        return replace(HasWorkers.action_components(self), building=self.building)

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        return None

    def invalid(
        self,
        resources: typing.Counter[Resource],
        dependencies: Dict[Building, Building],
        building_positions: BuildingPositions,
        *args,
        **kwargs,
    ) -> Optional[str]:
        dependency = dependencies[self.building]
        dependency_met = dependency in [*building_positions.values(), None]
        if not dependency_met:
            return f"Dependency ({dependency}) not met for {self}."
        insufficient_resources = Counter(self.building.cost) - resources
        if insufficient_resources:
            return "Insufficient resources"
        return None


@dataclass(frozen=True)
class BuildingCoordAction(HasWorkers, NoWorkersAction):
    building: Building
    coord: Coord

    def action_components(self) -> CompoundAction:
        return replace(
            HasWorkers.action_components(self), coord=self.coord, building=self.building
        )

    def assignment(self, positions: Positions) -> Assignment:
        i, j = astuple(self.coord)
        on_gas = Resource.GAS.on((i, j), positions)
        assimilator = isinstance(self.building, Assimilator)
        assert (on_gas and assimilator) or (not on_gas and not assimilator)
        assert not Resource.MINERALS.on((i, j), positions)
        return BuildOrder(self.building, (i, j))

    def get_workers(self) -> WorkerGenerator:
        for worker in self.workers:
            yield worker
            return  # assign at most one worker to building

    def invalid(
        self,
        resources: typing.Counter[Resource],
        dependencies: Dict[Building, Building],
        building_positions: BuildingPositions,
        pending_positions: BuildingPositions,
        positions: Positions,
    ) -> Optional[str]:
        dependency = dependencies[self.building]
        if dependency not in [*building_positions.values(), None]:
            return f"Dependency ({dependency}) not met for {self.building}."
        coord = astuple(self.coord)
        all_positions = {**building_positions, **pending_positions}
        if coord in all_positions:
            return f"coord occupied by {all_positions[coord]}"
        if isinstance(self.building, Assimilator):
            return (
                None
                if coord == positions[Resource.GAS]
                else f"Assimilator not built on gas"
            )
        else:
            return (
                "Building built on resource"
                if coord
                in (
                    positions[Resource.GAS],
                    positions[Resource.MINERALS],
                )
                else None
            )


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
    action: ActionStage
    building_positions: Dict[CoordType, Building]
    destroy: Dict[CoordType, Building]
    pointer: int
    positions: Dict[Union[Resource, Worker], CoordType]
    resources: typing.Counter[Resource]
    success: bool
    time_remaining: int
    valid: bool


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


def get_nearest(
    candidate_positions: List[CoordType],
    to: CoordType,
) -> CoordType:
    nearest = np.argmin(
        np.max(
            np.abs(
                np.expand_dims(np.array(to), 0) - np.stack(candidate_positions),
            ),
            axis=-1,
        )
    )
    return candidate_positions[int(nearest)]


class Assimilator(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=75, gas=0)

    @property
    def symbol(self) -> str:
        return "A"


class CyberneticsCore(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=0)

    @property
    def symbol(self) -> str:
        return "CC"


class DarkShrine(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=150)

    @property
    def symbol(self) -> str:
        return "DS"


class FleetBeacon(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=300, gas=200)

    @property
    def symbol(self) -> str:
        return "FB"


class Forge(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=0)

    @property
    def symbol(self) -> str:
        return "f"


class Gateway(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=0)

    @property
    def symbol(self) -> str:
        return "GW"


class Nexus(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=400, gas=0)

    @property
    def symbol(self) -> str:
        return "N"


class PhotonCannon(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=0)

    @property
    def symbol(self) -> str:
        return "PC"


class Pylon(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=100, gas=0)

    @property
    def symbol(self) -> str:
        return "P"


class RoboticsBay(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=200, gas=200)

    @property
    def symbol(self) -> str:
        return "RB"


class RoboticsFacility(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=200, gas=100)

    @property
    def symbol(self) -> str:
        return "RF"


class StarGate(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=150)

    @property
    def symbol(self) -> str:
        return "SG"


class TemplarArchives(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=200)

    @property
    def symbol(self) -> str:
        return "TA"


class TwilightCouncil(Building):
    @property
    def cost(self) -> Resources:
        return Resources(minerals=150, gas=100)

    @property
    def symbol(self) -> str:
        return "TC"


Buildings: List[Building] = [
    Assimilator(),
    CyberneticsCore(),
    DarkShrine(),
    FleetBeacon(),
    Forge(),
    Gateway(),
    Nexus(),
    PhotonCannon(),
    Pylon(),
    RoboticsBay(),
    RoboticsFacility(),
    StarGate(),
    TemplarArchives(),
    TwilightCouncil(),
]
WorldObjects = list(Buildings) + list(Resource) + list(Worker)
