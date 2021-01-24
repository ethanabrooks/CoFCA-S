import copy
import itertools
import pickle
import re
import sys
import typing
from collections import Counter, OrderedDict
from dataclasses import astuple, asdict, dataclass, field
from functools import lru_cache
from itertools import zip_longest
from multiprocessing import Queue
from pathlib import Path
from pprint import pprint
from queue import Full, Empty
from typing import Union, Dict, Generator, Tuple, List, Optional

import gym
import hydra
import numpy as np
from colored import fg
from gym import spaces
from gym.utils import seeding
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from treelib import Tree

import data_types
import keyboard_control
import osx_queue
from data_types import (
    Assignee,
    Unit,
    Units,
    ResourceCounter,
    UnitCounter,
    InitialAction,
    BuildOrder,
    Carrying,
    BuildingPositions,
    Assignment,
    Positions,
    CompoundAction,
    Obs,
    Resource,
    Building,
    WorldObject,
    WorldObjects,
    Worker,
    State,
    Line,
    ActionStage,
    RawAction,
    Buildings,
    Assimilator,
    Nexus,
)
from utils import RESET, Discrete

BuildingDependencies = Dict[Building, Building]
UnitDependencies = Dict[Unit, Building]


def multi_worker_symbol(num_workers: int):
    return f"w{num_workers}"


def strip_color(s: str):
    """
    https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", s)


@dataclass
class EnvConfig:
    attack_prob: float = 0
    break_on_fail: bool = False
    bucket_size: int = 5
    max_lines: int = 10
    min_lines: int = 2
    time_per_line: int = 4
    tgt_success_rate: float = 0.75
    world_size: int = 4


# noinspection PyAttributeOutsideInit
@dataclass
class Env(gym.Env):
    break_on_fail: bool
    bucket_size: int
    attack_prob: float
    failure_buffer: Queue
    max_lines: int
    min_lines: int
    rank: int
    random_seed: int
    tgt_success_rate: float
    time_per_line: int
    world_size: int
    alpha: float = 0.05
    evaluating: bool = None
    i: int = 0
    iterator = None
    render_thunk = None
    success_avg = 0.5
    success_with_failure_buf_avg = 0.5
    max_resources: ResourceCounter = field(
        default_factory=lambda: Counter({Resource.MINERALS: 500, Resource.GAS: 500})
    )

    def __post_init__(self):
        data_types.WORLD_SIZE = self.world_size
        self.random, _ = seeding.np_random(self.random_seed)
        self.n_lines_space = Discrete(self.min_lines, self.max_lines)
        self.n_lines_space.seed(self.random_seed)
        self.non_failure_random = self.random.get_state()
        action_components_space = CompoundAction.input_space()
        self.action_space = spaces.MultiDiscrete(
            [
                x
                for field in astuple(
                    RawAction(
                        delta=[2 * self.max_lines],
                        dg=[2],
                        ptr=[self.max_lines],
                        a=action_components_space.nvec,
                    )
                )
                for x in field
            ]
        )

        self.world_shape = world_shape = np.array([self.world_size, self.world_size])
        self.world_space = spaces.Box(
            low=np.zeros_like(world_shape, dtype=np.float32),
            high=(world_shape - 1).astype(np.float32),
        )

        channel_size = len(WorldObjects) + 1  # +1 for destroy
        max_shape = (channel_size, *world_shape)
        action_mask = spaces.MultiBinary(
            action_components_space.nvec.max() * action_components_space.nvec.size
        )
        destroyed_unit = spaces.Discrete(Unit.space().n + 1)  # +1 for None
        gate_openers = spaces.MultiDiscrete(
            np.array(
                [CompoundAction.input_space().nvec] * ActionStage.gate_opener_max_size()
            ).flatten()
        )
        instructions = spaces.MultiDiscrete(
            np.array(
                [Building.space().n + Unit.space().n + 1]  # +1 for padding
                * self.max_lines
            )
        )
        instruction_mask = spaces.MultiDiscrete(2 * np.ones(self.max_lines))
        obs = spaces.Box(
            low=np.zeros(max_shape, dtype=np.float32),
            high=np.ones(max_shape, dtype=np.float32),
        )
        partial_action = CompoundAction.representation_space()
        resources = spaces.MultiDiscrete([sys.maxsize] * 2)
        ptr = spaces.Discrete(self.max_lines)
        # noinspection PyTypeChecker
        self.obs_spaces = Obs(
            action_mask=action_mask,
            destroyed_unit=destroyed_unit,
            gate_openers=gate_openers,
            instructions=instructions,
            instruction_mask=instruction_mask,
            obs=obs,
            partial_action=partial_action,
            resources=resources,
            ptr=ptr,
        )
        self.observation_space = spaces.Dict(asdict(self.obs_spaces))

    def attack(
        self, building_positions: BuildingPositions, required: UnitCounter
    ) -> Tuple[Unit, BuildingPositions]:
        unit: Unit = self.random.choice(list(required))
        destructible: List[data_types.CoordType] = [
            c for c, b in building_positions.items() if not isinstance(b, Nexus)
        ]
        buildings = {}
        if destructible:
            num_destroyed = self.random.randint(len(destructible))
            destroy_idxs = self.random.choice(
                len(destructible), size=num_destroyed, replace=False
            )

            def get_buildings():
                for i in destroy_idxs:
                    destroy_coord = destructible[i]
                    yield destroy_coord, building_positions[destroy_coord]

            buildings = dict(get_buildings())

        return unit, buildings

    def build_building_dependencies(
        self, max_depth: int = None
    ) -> Generator[Tuple[Building, Optional[Building]], None, None]:
        buildings = [b for b in Buildings]
        self.random.shuffle(buildings)
        n = len(buildings)
        if max_depth is not None:
            n = min(max_depth, n)
        dependencies = np.round(self.random.random(n) * np.arange(n)) - 1
        dependencies = [None if i < 0 else buildings[int(i)] for i in dependencies]

        yield Assimilator(), None
        yield from itertools.zip_longest(buildings, dependencies)

    def build_instructions_and_dependencies(
        self,
    ) -> Tuple[List[Line], BuildingDependencies, UnitDependencies]:
        assert (
            self.n_lines_space.low >= 2
        ), "At least 2 lines required to build a worker."

        building_dependencies: BuildingDependencies = dict(
            self.build_building_dependencies()
        )
        unit_dependencies: UnitDependencies = dict(self.build_unit_dependencies())

        def instructions_for(building: Building):
            if building is None:
                return
            yield from instructions_for(building_dependencies[building])
            yield building

        n_lines = self.n_lines_space.sample()
        instructions_for = {b: [*instructions_for(b)] for b in Buildings}

        def random_instructions(n):
            if n <= 0:
                return
            _instructions_for = {
                b: i for b, i in instructions_for.items() if len(i) < n
            }
            possible_units = [
                u for u, b in unit_dependencies.items() if b in _instructions_for.keys()
            ]
            if not possible_units:
                return
            unit = possible_units[self.random.choice(len(possible_units))]
            building = unit_dependencies[unit]
            _instructions = [*_instructions_for[building], unit]
            yield from _instructions
            yield from random_instructions(n - len(_instructions))

        instructions = [*random_instructions(n_lines)]
        if not instructions:
            return self.build_instructions_and_dependencies()
        assert len(instructions) >= 2
        return instructions, building_dependencies, unit_dependencies

    @staticmethod
    def build_trees(dependencies: BuildingDependencies) -> typing.Set[Tree]:

        trees: Dict[Building, Tree] = {}

        def create_nodes(bldg: Building):
            if bldg in trees:
                return
            dependency = dependencies[bldg]
            if dependency is None:
                trees[bldg] = Tree()
            else:
                create_nodes(dependency)
                trees[bldg] = trees[dependency]
            trees[bldg].create_node(str(bldg), bldg, parent=dependency)

        for building in Buildings:
            create_nodes(building)
        return set(trees.values())

    def build_unit_dependencies(
        self,
    ) -> Generator[Tuple[Unit, Building], None, None]:
        buildings = [
            Building.parse(i)
            for i in self.random.choice(len(Buildings), size=len(Units))
        ]
        units = copy.copy(Units)
        self.random.shuffle(units)
        yield from zip(units, buildings)

    @staticmethod
    def done_generator():
        state: State
        state = yield

        while True:
            # noinspection PyTypeChecker
            state = (
                yield state.success or not state.time_remaining,
                lambda: None,
            )

    @staticmethod
    def dump(name: str, x) -> Path:
        path = Path(f"{name}.pkl")
        with path.open("wb") as f:
            pickle.dump(x, f)
        return path.absolute()

    def failure_buffer_wrapper(self, iterator):
        use_failure_buf = False
        size = self.failure_buffer.qsize()
        if self.evaluating or not size:
            use_failure_buf = False
        else:
            success_avg = max(
                self.success_avg, self.success_with_failure_buf_avg + 1e-6
            )
            tgt_success_rate = max(
                self.success_with_failure_buf_avg,
                min(self.tgt_success_rate, success_avg),
            )
            use_failure_prob = 1 - (
                tgt_success_rate - self.success_with_failure_buf_avg
            ) / (success_avg - self.success_with_failure_buf_avg)
            use_failure_buf = self.random.random() < use_failure_prob
        state = None
        if use_failure_buf:

            # randomly rotate queue
            for i in range(self.random.choice(min(100, size))):
                try:
                    state = self.failure_buffer.get_nowait()
                    self.failure_buffer.put_nowait(state)
                except Full:
                    pass  # discard, keep going
                except Empty:
                    break

            try:
                state = self.failure_buffer.get_nowait()
            except (Full, Empty):
                use_failure_buf = state is not None

        if not use_failure_buf:
            state = self.non_failure_random
        self.random.set_state(state)
        initial_random = self.random.get_state()
        action = None

        def render_thunk():
            return

        def render():
            render_thunk()
            if use_failure_buf:
                print(fg("red"), "Used failure buffer", RESET)
            else:
                print(fg("blue"), "Did not use failure buffer", RESET)

        while True:
            s, r, t, i = iterator.send(action)
            render_thunk = self.render_thunk
            self.render_thunk = render
            if t:
                success = i["success"]

                if not self.evaluating:
                    i.update(
                        {
                            f"{k} ({'with' if use_failure_buf else 'without'} failure buffer)": v
                            for k, v in i.items()
                        }
                    )

                def interpolate(old, new):
                    return old + self.alpha * (new - old)

                if use_failure_buf:
                    self.success_with_failure_buf_avg = interpolate(
                        self.success_with_failure_buf_avg, success
                    )
                else:
                    self.success_avg = interpolate(self.success_avg, success)

                put_failure_buf = not self.evaluating and not success
                if put_failure_buf:
                    try:
                        self.failure_buffer.put_nowait(initial_random)
                    except Full:
                        pass

                i.update({"used failure buffer": use_failure_buf})

            if t:
                # noinspection PyAttributeOutsideInit
                self.non_failure_random = self.random.get_state()
            action = yield s, r, t, i

    def info_generator(self, *lines):
        state: State
        done: bool
        state, done = yield
        info = {}
        elapsed_time = -1

        while True:
            if done:
                if self.evaluating:
                    lower = (len(lines) - 1) // self.bucket_size * self.bucket_size + 1
                    upper = (
                        1 + (len(lines) - 1) // self.bucket_size
                    ) * self.bucket_size
                    key = (
                        f"success on instructions length-{lower} through length-{upper}"
                    )
                else:
                    key = f"success on length-{len(lines)} instructions"
                info.update(
                    {
                        f"success": float(state.success),
                        key: float(state.success),
                        "instruction length": len(lines),
                        "time per line": elapsed_time / len(lines),
                    },
                )

            # noinspection PyTupleAssignmentBalance
            state, done = yield info, lambda: None
            info = {}
            elapsed_time += 1

    @staticmethod
    def load(path: str) -> State:
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def main(self):
        keyboard_control.run(self, lambda: None)

    def obs_generator(
        self,
        *lines: Line,
        building_dependencies: BuildingDependencies,
        unit_dependencies: UnitDependencies,
    ):
        state: State
        state = yield

        padded: List[Optional[Line]] = [
            *lines,
            *[None] * (self.max_lines - len(lines)),
        ]

        def render():
            pprint(unit_dependencies)

            def buildings_required_for(
                unit_or_building: Union[Unit, Building]
            ) -> Generator[Building, None, None]:
                if isinstance(unit_or_building, Unit):
                    building = unit_dependencies[unit_or_building]
                elif isinstance(unit_or_building, Building):
                    building = building_dependencies[unit_or_building]
                else:
                    raise RuntimeError
                if building not in [None, *state.building_positions.values()]:
                    yield building
                    yield from buildings_required_for(building)

            def get_required_buildings():
                for unit in state.required_units:
                    yield from buildings_required_for(unit)

            required_buildings = set(get_required_buildings())

            for i, line in enumerate(lines):
                print(
                    "{:2}{}{} {}".format(
                        i,
                        "-" if i == state.pointer else " ",
                        "*" if line in required_buildings else " ",
                        repr(line),
                    )
                )
            print("Obs:")
            for string in self.room_strings(world):
                print(string, end="")

        def coords():
            yield from state.positions.items()
            for p, b in state.building_positions.items():
                yield b, p

        while True:
            world = np.zeros(self.obs_spaces.obs.shape)
            for o, p in coords():
                world[(WorldObjects.index(o), *p)] = 1
            for p in state.destroyed_buildings.keys():
                world[(-1, *p)] = 1
            assert isinstance(state.action, ActionStage)

            gate_openers: np.ndarray = self.obs_spaces.gate_openers.nvec.copy().reshape(
                -1, CompoundAction.input_space().nvec.size
            )
            gate_openers -= 1
            unpadded_gate_openers = state.action.gate_openers()
            gate_openers[: len(unpadded_gate_openers)] = unpadded_gate_openers

            destroyed_unit = (
                0 if state.destroyed_unit is None else 1 + state.destroyed_unit.to_int()
            )
            obs = OrderedDict(
                asdict(
                    Obs(
                        action_mask=state.action.mask(unit_dependencies).ravel(),
                        destroyed_unit=destroyed_unit,
                        gate_openers=gate_openers.ravel(),
                        instruction_mask=(np.array([int(p is None) for p in padded])),
                        instructions=(np.array([*map(self.preprocess_line, padded)])),
                        obs=world,
                        partial_action=(np.array([*state.action.to_ints()])),
                        ptr=state.pointer,
                        resources=(np.array([state.resources[r] for r in Resource])),
                    )
                )
            )
            for (k, space), (n, o) in zip(
                self.observation_space.spaces.items(), obs.items()
            ):
                if not space.contains(o):
                    import ipdb

                    ipdb.set_trace()
                    space.contains(o)
            # noinspection PyTypeChecker
            state = yield obs, lambda: render()  # perform time-step

    def place_objects(
        self, n_lines: int
    ) -> Generator[Tuple[WorldObject, np.ndarray], None, None]:
        nexus = self.random.choice(self.world_size, size=2)
        yield Nexus(), nexus
        for w in Worker:
            yield w, nexus
        resource_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        resource_locations = [
            *filter(
                self.world_space.contains,
                nexus + resource_offsets,
            )
        ]
        minerals, gas = self.random.choice(
            len(resource_locations), size=2, replace=False
        )
        minerals = resource_locations[minerals]
        gas = resource_locations[gas]
        yield Resource.MINERALS, minerals
        yield Resource.GAS, gas

        occupied = [nexus, minerals, gas]
        occupied_indices = np.sort(
            np.ravel_multi_index(np.stack(occupied, axis=-1), self.world_shape)
        )

        max_initial_buildings = max(0, (self.world_size ** 2 - len(occupied) - n_lines))
        if max_initial_buildings > 0:
            num_initial_buildings = self.random.randint(max_initial_buildings + 1)
            initial_index = self.random.choice(
                self.world_size ** 2 - len(occupied),
                size=num_initial_buildings,
                replace=False,
            )
            for i in occupied_indices:
                initial_index[initial_index >= i] += 1
            initial_pos = np.stack(
                np.unravel_index(initial_index, self.world_shape), axis=-1
            )
            initial_buildings = self.random.choice(
                Buildings,
                size=num_initial_buildings,
            )
            for b, p in zip(initial_buildings, initial_pos):
                # assert not any(np.array_equal(p, p_) for p_ in occupied)
                # occupied += [p]
                yield b, gas if isinstance(b, Assimilator) else p

    @staticmethod
    def preprocess_line(line: Optional[Line]) -> int:
        if line is None:
            return 0
        if isinstance(line, Building):
            return 1 + line.to_int()
        if isinstance(line, Unit):
            return 1 + Building.space().n + line.to_int()
        raise RuntimeError

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    def reset(self):
        self.i += 1
        self.iterator = self.srti_generator()
        s, r, t, i = next(self.iterator)
        return s

    def room_strings(self, room):
        max_symbol_size = max(
            [
                len(multi_worker_symbol(len(Worker))),
                *[len(strip_color(str(x.symbol))) for x in WorldObjects],
            ]
        )
        max_symbols_per_grid = 3
        for i, row in enumerate(room.transpose((1, 2, 0)).astype(int)):
            for j, channel in enumerate(row):
                (nonzero,) = channel[: len(WorldObjects)].nonzero()
                objects = [WorldObjects[k] for k in nonzero]
                worker_symbol = None
                if len(objects) > max_symbols_per_grid:
                    worker_symbol = f"w{sum([isinstance(o, Worker) for o in objects])}"
                    objects = [o for o in objects if not isinstance(o, Worker)]
                symbols = [o.symbol for o in objects]
                if worker_symbol is not None:
                    symbols += [worker_symbol]

                for _, symbol in zip_longest(range(max_symbols_per_grid), symbols):
                    if symbol is None:
                        symbol = " " * max_symbol_size
                    else:
                        symbol += " " * (max_symbol_size - len(strip_color(symbol)))
                    yield from symbol
                yield RESET
                yield "|"
            grid_size = max_symbols_per_grid * max_symbol_size
            yield f"\n" + ("-" * (grid_size) + "+") * self.world_size + "\n"

    @staticmethod
    def reward_generator():
        state: State
        state = yield

        while True:
            reward = float(state.success)
            # noinspection PyTypeChecker
            state = yield reward, lambda: print("Reward:", reward)

    def seed(self, seed=None):
        assert self.random_seed == seed

    def srti_generator(
        self,
    ) -> Generator[Tuple[any, float, bool, dict], Optional[RawAction], None]:
        (
            instructions,
            building_dependencies,
            unit_dependencies,
        ) = self.build_instructions_and_dependencies()
        assert len(instructions) >= 2
        obs_iterator = self.obs_generator(
            *instructions,
            building_dependencies=building_dependencies,
            unit_dependencies=unit_dependencies,
        )
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator(*instructions)
        state_iterator = self.state_generator(
            *instructions,
            building_dependencies=building_dependencies,
            unit_dependencies=unit_dependencies,
        )
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        state, render_state = next(state_iterator)

        def render():
            # for tree in self.build_trees(building_dependencies):
            #     tree.show()

            if t:
                print(fg("green") if i["success"] else fg("red"))
            render_r()
            render_t()
            render_i()
            render_state()
            render_s()
            print(RESET)

        while True:
            s, render_s = obs_iterator.send(state)
            r, render_r = reward_iterator.send(state)
            t, render_t = done_iterator.send(state)
            i, render_i = info_iterator.send((state, t))

            if self.break_on_fail and t and not i["success"]:
                import ipdb

                ipdb.set_trace()

            self.render_thunk = render

            a: Optional[RawAction]
            # noinspection PyTypeChecker
            a = yield s, r, t, i

            state, render_state = state_iterator.send(a)

    def state_generator(
        self,
        *instructions: Line,
        building_dependencies: Dict[Building, Building],
        unit_dependencies: Dict[Unit, Building],
    ) -> Generator[State, Optional[RawAction], None]:
        positions: List[Tuple[WorldObject, np.ndarray]] = [
            *self.place_objects(len(instructions))
        ]
        building_positions: BuildingPositions = dict(
            [((i, j), b) for b, (i, j) in positions if isinstance(b, Building)]
        )
        positions: Positions = dict(
            [
                (o, (i, j))
                for o, (i, j) in positions
                if isinstance(o, (Resource, Worker))
            ]
        )
        assignments: Dict[Assignee, Assignment] = {w: Resource.MINERALS for w in Worker}
        initial_required: UnitCounter = Counter(
            l for l in instructions if isinstance(l, Unit)
        )
        deployed: UnitCounter = Counter()
        resources: ResourceCounter = Counter(
            {
                r: round(self.random.randint(v) / 25) * 25
                for r, v in self.max_resources.items()
            }
        )
        carrying: Carrying = {w: None for w in Worker}
        ptr: int = 0
        destroyed_unit: Optional[Unit] = None
        destroyed_buildings: BuildingPositions = {}
        action = InitialAction()
        time_remaining = (1 + len(instructions)) * self.time_per_line
        error_msg = None
        pending_costs = Counter

        def render():
            print("Time remaining:", time_remaining)
            print("Required:")
            pprint(required)
            if pending_costs:
                print("Pending costs:")
                pprint(pending_costs)
            print()
            pprint(action if error_msg is None else new_action)
            print()
            for k, v in sorted(assignments.items()):
                print(f"{k}: {v}")
            if destroyed_unit:
                print(fg("red"), "Destroyed unit:", destroyed_unit, RESET)
            if destroyed_buildings:
                print(fg("red"), "Destroyed buildings:", sep="")
                pprint(destroyed_buildings)
                print(RESET, end="")
            if error_msg is not None:
                print(fg("red"), error_msg, RESET, sep="")

        self.render_thunk = render

        while True:
            resources = resources & self.max_resources  # cap resources
            required = initial_required - deployed
            success = not required

            pending_positions = {
                a.coord: a.building
                for a in assignments.values()
                if isinstance(a, BuildOrder)
            }
            pending_costs = Counter(
                [r for b in pending_positions.values() for r in b.cost]
            )

            state = State(
                building_positions=building_positions,
                destroyed_buildings=destroyed_buildings,
                destroyed_unit=destroyed_unit,
                pending_positions=pending_positions,
                positions=positions,
                required_units=required,
                resources=resources,
                success=success,
                pointer=ptr,
                action=action,
                time_remaining=time_remaining,
                valid=error_msg is None,
            )

            raw_action: Optional[RawAction]
            # noinspection PyTypeChecker
            raw_action = yield state, render
            if raw_action is None:
                new_action = action.from_input(building_positions)
            elif isinstance(raw_action, RawAction):
                a, ptr = map(int, raw_action.a), int(raw_action.ptr)
                new_action = action.update(*a, building_positions=building_positions)
            else:
                raise RuntimeError

            error_msg = new_action.invalid(
                resources=resources,
                dependencies=building_dependencies,
                building_positions=building_positions,
                pending_costs=pending_costs,
                pending_positions=pending_positions,
                positions=positions,
                unit_dependencies=unit_dependencies,
            )
            if error_msg is not None:
                time_remaining -= 1  # penalize agent for invalid
                continue

            action = new_action
            assignment = action.assignment(positions)
            is_op = assignment is not None
            if is_op:
                time_remaining -= 1
            else:
                continue

            for worker in action.assignee():
                assignments[worker] = assignment

            assignee: Assignee
            for (assignee, assignment,) in sorted(
                assignments.items(),
                key=lambda t: isinstance(t[1], Resource),
                reverse=True,
            ):  # collect resources first.
                assignment: Assignment
                assignment.execute(
                    assignee=assignee,
                    assignments=assignments,
                    building_positions=building_positions,
                    carrying=carrying,
                    deployed_units=deployed,
                    pending_costs=pending_costs,
                    pending_positions=pending_positions,
                    positions=positions,
                    resources=resources,
                )

            if self.random.random() < self.attack_prob / len(instructions):
                destroyed_unit, destroyed_buildings = self.attack(
                    building_positions, required
                )
                required.subtract([destroyed_unit])
                for coord in destroyed_buildings.keys():
                    del building_positions[coord]

    def step(self, action: Union[np.ndarray, ActionStage]):
        if isinstance(action, np.ndarray):
            action = RawAction.parse(*action)
        return self.iterator.send(action)


@hydra.main(config_name="config")
def app(cfg: DictConfig) -> None:
    failure_buffer = Queue()
    try:
        failure_buffer.qsize()
    except NotImplementedError:
        failure_buffer = osx_queue.Queue()
    Env(
        **cfg,
        rank=0,
        failure_buffer=failure_buffer,
    ).main()


if __name__ == "__main__":

    @dataclass
    class Config(EnvConfig):
        random_seed: int = 0

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    app()
