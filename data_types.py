import typing
from abc import abstractmethod, ABC
from collections import Counter
from dataclasses import dataclass, astuple, field, asdict
from enum import unique, Enum, auto
from typing import Tuple, Union, List, Generator, Dict, Generic, Optional

import numpy as np
import torch
from colored import fg, sys
from gym import Space, spaces

from utils import RESET

CoordType = Tuple[int, int]

WORLD_SIZE = None


def move_from(origin: CoordType, toward: CoordType) -> CoordType:
    origin = np.array(origin)
    return tuple(
        origin
        + np.clip(
            np.array(toward) - origin,
            -1,
            1,
        )
    )


""" abstract classes """


class WorldObject:
    @property
    @abstractmethod
    def symbol(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class ActionComponent:
    @staticmethod
    @abstractmethod
    def parse(n: int) -> "ActionComponent":
        pass

    @staticmethod
    @abstractmethod
    def input_space() -> spaces.Discrete:
        pass

    @staticmethod
    @abstractmethod
    def representation_space() -> spaces.MultiDiscrete:
        pass

    @abstractmethod
    def to_ints(self) -> Generator[int, None, None]:
        pass

    @staticmethod
    @abstractmethod
    def zeros() -> Generator[int, None, None]:
        pass


class Building(WorldObject, ActionComponent, ABC):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type)

    def __str__(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def cost(self) -> "Resources":
        pass

    def on(self, coord: "CoordType", building_positions: "BuildingPositions"):
        return self == building_positions.get(coord)

    @staticmethod
    def parse(n: int) -> ActionComponent:
        return Buildings[n]

    @staticmethod
    def input_space() -> spaces.Discrete:
        return spaces.Discrete(len(Buildings))

    @staticmethod
    def representation_space() -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([len(Buildings)])

    @property
    @abstractmethod
    def symbol(self) -> str:
        pass

    def to_ints(self) -> Generator[int, None, None]:
        yield Buildings.index(self)

    @staticmethod
    def zeros() -> Generator[int, None, None]:
        yield 0


class Assignment:
    @abstractmethod
    def execute(
        self,
        assignments: "Assignments",
        building_positions: "BuildingPositions",
        pending_positions: "BuildingPositions",
        carrying: "Carrying",
        positions: "Positions",
        resources: typing.Counter["Resource"],
        worker: "Worker",
    ) -> None:
        raise NotImplementedError


""" world objects"""


@unique
class Worker(WorldObject, Enum):
    A = auto()
    B = auto()
    C = auto()

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

    @property
    def symbol(self):
        return self.value


@unique
class Resource(WorldObject, Assignment, Enum):
    MINERALS = auto()
    GAS = auto()

    def __hash__(self):
        return Enum.__hash__(self)

    def __eq__(self, other):
        return Enum.__eq__(self, other)

    def execute(
        self,
        worker: Worker,
        positions: "Positions",
        building_positions: "BuildingPositions",
        carrying: "Carrying",
        resources: typing.Counter["Resource"],
        **kwargs,
    ) -> None:
        worker_pos = positions[worker]

        if carrying[worker] is None:
            resource_pos = positions[self]
            positions[worker] = tuple(move_from(worker_pos, toward=resource_pos))
            worker_pos = positions[worker]
            if worker_pos == resource_pos:
                if self is Resource.GAS and not isinstance(
                    building_positions.get(positions[worker]), Assimilator
                ):
                    return  # no op on gas unless Assimilator
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
                assert isinstance(carrying[worker], Resource)
                resources[carrying[worker]] += 1
                carrying[worker] = None

    def on(
        self,
        coord: "CoordType",
        positions: "Positions",
    ) -> bool:
        return positions[self] == coord

    @property
    def symbol(self):
        if self is Resource.GAS:
            return fg("green") + "G" + RESET
        if self is Resource.MINERALS:
            return fg("blue") + "M" + RESET
        raise RuntimeError


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
class Workers(ActionComponent):
    workers: typing.Set[Worker] = field(default_factory=lambda: set())

    def __iter__(self):
        yield from self.workers

    @staticmethod
    def parse(n: int) -> "Workers":
        membership = np.unravel_index(n, [2 for _ in Worker])
        return Workers({w for w, m in zip(Worker, membership) if m})

    @staticmethod
    def input_space() -> spaces.Discrete:
        return spaces.Discrete(2 ** len(Worker))

    @staticmethod
    def representation_space() -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete(2 * np.ones(len(Worker)))

    def to_ints(self) -> Generator[int, None, None]:
        for worker in Worker:
            yield int(worker in self.workers)

    @staticmethod
    def zeros() -> Generator[int, None, None]:
        for _ in Worker:
            yield 0


@dataclass
class Coord(ActionComponent):
    i: int
    j: int

    @staticmethod
    def parse(n: int) -> "ActionComponent":
        assert isinstance(WORLD_SIZE, int)
        ij = np.unravel_index(n, (WORLD_SIZE, WORLD_SIZE))
        return Coord(*ij)

    @staticmethod
    def input_space() -> spaces.Discrete:
        assert isinstance(WORLD_SIZE, int)
        return spaces.Discrete(WORLD_SIZE ** 2)

    @staticmethod
    def representation_space() -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([WORLD_SIZE, WORLD_SIZE])

    def to_ints(self) -> Generator[int, None, None]:
        yield self.i
        yield self.j

    @staticmethod
    def zeros() -> Generator[int, None, None]:
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

    def execute(
        self,
        assignments: Assignments,
        worker: Worker,
        positions: Positions,
        building_positions: BuildingPositions,
        pending_positions: "BuildingPositions",
        resources: typing.Counter[Resource],
        **kwargs,
    ) -> None:
        if positions[worker] == self.coord:
            building_positions[self.coord] = self.building
            assignments[worker] = DoNothing()
        else:
            if self.coord not in pending_positions:
                pending_positions[self.coord] = self.building
                resources.subtract(self.building.cost)
            GoTo(self.coord).execute(
                assignments=assignments,
                worker=worker,
                positions=positions,
                building_positions=building_positions,
                **kwargs,
            )


@dataclass(frozen=True)
class GoTo(Assignment):
    coord: CoordType

    def execute(self, positions: "Positions", worker: "Worker", **kwargs) -> None:
        positions[worker] = move_from(positions[worker], toward=self.coord)


class DoNothing(Assignment):
    def execute(self, *args, **kwargs) -> None:
        pass


Command = Union[BuildOrder, Resource]

O = typing.TypeVar("O", Space, torch.Tensor, np.ndarray, int)


@dataclass(frozen=True)
class Obs(typing.Generic[O]):
    action_mask: O
    can_open_gate: O
    line_mask: O
    lines: O
    obs: O
    partial_action: O
    ptr: O
    resources: O


X = typing.TypeVar("X")


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
    @abstractmethod
    def _parse_int(self, a: int) -> ActionComponent:
        pass

    @staticmethod
    @abstractmethod
    def _parse_string(s: str) -> ActionComponent:
        pass

    @staticmethod
    @abstractmethod
    def _prompt() -> str:
        pass

    @abstractmethod
    def _update(self, a: ActionComponent) -> "CompoundAction":
        pass

    @abstractmethod
    def active_components(self) -> Generator[type, None, None]:
        pass

    @abstractmethod
    def assignment(self, positions: Positions) -> Optional[Assignment]:
        pass

    def can_open_gate(self) -> Generator[bool, None, None]:
        for a, mask in enumerate(self.mask()):
            if mask:
                yield False
                continue
            try:
                yield isinstance(self.update(a), NoWorkersAction)
            except (IndexError, ValueError):
                yield False

    def from_input(self) -> ActionComponent:
        while True:
            try:
                return self._parse_string(input(self._prompt() + "\n"))
            except ValueError as e:
                print(e)

    def get_building(self) -> Optional[Building]:
        return None

    def get_coord(self) -> Optional[Coord]:
        return None

    def get_workers(self) -> Optional[Workers]:
        return None

    def components(self) -> Generator[ActionComponent, None, None]:
        yield self.get_workers()
        yield self.get_building()
        yield self.get_coord()

    @staticmethod
    def component_classes() -> Generator[type, None, None]:
        yield Workers
        yield Building
        yield Coord

    @staticmethod
    def compound_actions() -> Generator["CompoundAction", None, None]:
        yield NoWorkersAction
        yield WorkersAction
        yield CoordAction
        yield BuildingAction
        yield BuildingCoordAction

    @classmethod
    def space(cls) -> spaces.MultiDiscrete:
        def gen():
            for component in cls.component_classes():
                assert issubclass(component, ActionComponent)
                yield from 1 + component.representation_space().nvec

        return spaces.MultiDiscrete([*gen()])

    def active(self):
        active_components = {*self.active_components()}
        for component in self.component_classes():
            assert issubclass(component, ActionComponent)
            active = component in active_components
            yield component, active

    def mask(self) -> Generator[int, None, None]:
        for component, active in self.active():
            for a in range(component.input_space().n):
                yield not active

    @classmethod
    def action_size(cls) -> int:
        def gen():
            for c in cls.component_classes():
                assert issubclass(c, ActionComponent)
                yield c.input_space().n

        return sum(gen())

    def to_ints(self) -> Generator[int, None, None]:
        for cls, component in zip(self.component_classes(), self.components()):
            if component is None:
                assert issubclass(cls, ActionComponent)
                yield from cls.zeros()
            else:
                assert isinstance(component, ActionComponent)
                for n in component.to_ints():
                    yield n + 1

    def update(self, a: Union[int, ActionComponent]) -> "CompoundAction":
        if isinstance(a, int):
            for component, active in self.active():
                if active:
                    a = self._parse_int(a)
                    break
                else:
                    a -= component.input_space().n
        return self._update(a)

    @abstractmethod
    def valid(
        self,
        resources: typing.Counter[Resource],
        dependencies: Dict[Building, Building],
        building_positions: BuildingPositions,
        pending_positions: BuildingPositions,
        positions: Positions,
    ) -> bool:
        pass


@dataclass(frozen=True)
class NoWorkersAction(CompoundAction):
    def _parse_int(self, a: int) -> Workers:
        return Workers.parse(a)

    @staticmethod
    def _parse_string(s: str) -> ActionComponent:
        return Workers({Worker(int(i)) for i in s})

    @staticmethod
    def _prompt() -> str:
        return "Workers:"

    def _update(self, a: ActionComponent) -> Union["WorkersAction", "NoWorkersAction"]:
        if not [*a]:
            return NoWorkersAction()
        assert isinstance(a, Workers)
        return WorkersAction(a)

    def active_components(self) -> Generator[type, None, None]:
        yield Workers

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        return DoNothing()

    def get_workers(self) -> Optional[Workers]:
        return Workers()

    def valid(self, *args, **kwargs) -> bool:
        return True


def parse_coord(s):
    i, j = map(int, s.split())
    assert 0 <= i < WORLD_SIZE
    assert 0 <= j < WORLD_SIZE
    return Coord(i, j)


@dataclass(frozen=True)
class WorkersAction(CompoundAction):
    workers: Workers

    def _parse_int(self, a: int) -> ActionComponent:
        try:
            return Coord.parse(a)
        except ValueError:
            return Buildings[a - Coord.input_space().n]

    @staticmethod
    def _parse_string(s: str) -> ActionComponent:
        try:
            return Buildings[int(s)]
        except ValueError:
            return parse_coord(s)

    def _prompt(self) -> str:
        return "\n".join(
            [f"({i}) {b}" for i, b in enumerate(Buildings)] + ["Coord or Building"]
        )

    def _update(self, a: ActionComponent) -> "CompoundAction":
        if isinstance(a, Coord):
            return CoordAction(workers=self.workers, coord=a)
        elif isinstance(a, Building):
            return BuildingAction(workers=self.workers, building=a)
        else:
            raise RuntimeError

    def active_components(self) -> Generator[type, None, None]:
        yield Coord
        yield Building

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        return None

    def get_workers(self) -> Optional[Workers]:
        return self.workers

    def valid(self, *args, **kwargs) -> bool:
        return True


@dataclass(frozen=True)
class CoordAction(NoWorkersAction):
    workers: Workers
    coord: Coord

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        for resource in Resource:
            if resource.on(astuple(self.coord), positions):
                return resource
        return GoTo(astuple(self.coord))

    def get_workers(self) -> Optional[Workers]:
        return self.workers

    def get_coord(self) -> Optional[Coord]:
        return self.coord

    def valid(self, *args, **kwargs) -> bool:
        return True


@dataclass(frozen=True)
class BuildingAction(CompoundAction):
    workers: Workers
    building: Building

    def _parse_int(self, a: int) -> ActionComponent:
        return Coord.parse(a)

    def _parse_string(self, s: str) -> ActionComponent:
        return parse_coord(s)

    def _prompt(self) -> str:
        return "Coord"

    def _update(self, a: Coord) -> "BuildingCoordAction":
        return BuildingCoordAction(
            workers=self.workers, building=self.building, coord=a
        )

    def active_components(self) -> Generator[type, None, None]:
        yield Coord

    def assignment(self, positions: Positions) -> Optional[Assignment]:
        return None

    def get_workers(self) -> Optional[Workers]:
        return self.workers

    def get_building(self) -> Optional[Building]:
        return self.building

    def valid(
        self,
        resources: typing.Counter[Resource],
        dependencies: Dict[Building, Building],
        building_positions: BuildingPositions,
        **kwargs,
    ) -> bool:
        dependency = dependencies[self.building]
        dependency_met = dependency in [*building_positions.values(), None]
        insufficient_resources = Counter(self.building.cost) - resources
        return dependency_met and not insufficient_resources


@dataclass(frozen=True)
class BuildingCoordAction(NoWorkersAction):
    workers: Workers
    building: Building
    coord: Coord

    def assignment(self, positions: Positions) -> Assignment:
        coord = astuple(self.coord)
        on_gas = Resource.GAS.on(coord, positions)
        assimilator = isinstance(self.building, Assimilator)
        assert (on_gas and assimilator) or (not on_gas and not assimilator)
        assert not Resource.MINERALS.on(coord, positions)
        return BuildOrder(self.building, coord)

    def get_building(self) -> Optional[Building]:
        return self.building

    def get_coord(self) -> Optional[Coord]:
        return self.coord

    def get_workers(self) -> Optional[Workers]:
        return self.workers

    def valid(
        self,
        resources: typing.Counter[Resource],
        dependencies: Dict[Building, Building],
        building_positions: BuildingPositions,
        pending_positions: BuildingPositions,
        positions: Positions,
    ) -> bool:
        coord = astuple(self.coord)
        if coord in {**building_positions, **pending_positions}:
            return False
        if isinstance(self.building, Assimilator):
            return coord == positions[Resource.GAS]
        else:
            return coord not in (
                positions[Resource.GAS],
                positions[Resource.MINERALS],
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
    action: CompoundAction
    building_positions: Dict[CoordType, Building]
    pointer: int
    positions: Dict[Union[Resource, Worker], CoordType]
    resources: typing.Counter[Resource]
    success: bool
    time_remaining: int


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
