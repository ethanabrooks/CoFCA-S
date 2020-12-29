import itertools
import re
import sys
import typing
from collections import Counter, OrderedDict
from dataclasses import astuple, asdict, dataclass, replace
from itertools import zip_longest
from multiprocessing import Queue
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
    NoWorkersAction,
    Carrying,
    BuildingPositions,
    Assignment,
    Positions,
    ActionComponent,
    Obs,
    Resource,
    Building,
    WorldObject,
    WorldObjects,
    Worker,
    State,
    Line,
    CompoundAction,
    RawAction,
    Buildings,
    Assimilator,
    Nexus,
)
from utils import RESET, Discrete, get_max_shape

Dependencies = Dict[Building, Building]


def multi_worker_symbol(num_workers: int):
    return f"w{num_workers}"


def strip_color(s: str):
    """
    https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", s)


@dataclass
class EnvConfig:
    break_on_fail: bool = False
    attack_prob: float = 0
    max_lines: int = 10
    min_lines: int = 1
    num_initial_buildings: int = 2
    time_per_line: int = 4
    tgt_success_rate: float = 0.75
    world_size: int = 3


# noinspection PyAttributeOutsideInit
@dataclass
class Env(gym.Env):
    break_on_fail: bool
    attack_prob: float
    eval_steps: int
    failure_buffer: Queue
    max_lines: int
    min_lines: int
    num_initial_buildings: int
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

    def __post_init__(self):
        super().__init__()
        data_types.WORLD_SIZE = self.world_size
        self.random, _ = seeding.np_random(self.random_seed)
        self.n_lines_space = Discrete(self.min_lines, self.max_lines)
        self.n_lines_space.seed(self.random_seed)
        self.non_failure_random = self.random.get_state()
        action_space = CompoundAction.input_space()
        self.action_space = spaces.MultiDiscrete(
            [
                x
                for field in astuple(
                    RawAction(
                        delta=[2 * self.max_lines],
                        dg=[2],
                        ptr=[self.max_lines],
                        a=CompoundAction.input_space().nvec,
                    )
                )
                for x in field
            ]
        )

        lines_space = spaces.MultiDiscrete(
            np.array([[2, len(Buildings)]] * self.max_lines)
        )
        line_mask_space = spaces.MultiDiscrete(2 * np.ones(self.max_lines))
        self.world_shape = world_shape = np.array([self.world_size, self.world_size])
        self.world_space = spaces.Box(
            low=np.zeros_like(world_shape, dtype=np.float32),
            high=(world_shape - 1).astype(np.float32),
        )

        max_shape = (len(WorldObjects), *world_shape)
        obs_space = spaces.Box(
            low=np.zeros(max_shape, dtype=np.float32),
            high=np.ones(max_shape, dtype=np.float32),
        )
        resources_space = spaces.MultiDiscrete([sys.maxsize] * 2)
        pointer_space = spaces.Discrete(self.max_lines)

        # noinspection PyTypeChecker
        def gate_openers():
            for subclass in CompoundAction.subclasses():
                yield subclass.gate_openers()

        gate_openers = [np.array(o) for o in gate_openers()]
        self.gate_opener_shape = get_max_shape(*gate_openers)
        padded = np.stack([self.pad_gate_openers(o) for o in gate_openers])
        assert np.all(np.min(padded, axis=0) == 0)
        gate_opener_space = spaces.MultiDiscrete(
            np.max(padded, axis=0) + 1
        )  # +1 because upper bound is not inclusive
        action_mask_space = spaces.MultiBinary(
            action_space.nvec.max() * action_space.nvec.size
        )
        self.observation_space = spaces.Dict(
            asdict(
                Obs(
                    action_mask=action_mask_space,
                    gate_openers=gate_opener_space,
                    lines=lines_space,
                    line_mask=line_mask_space,
                    obs=obs_space,
                    partial_action=CompoundAction.representation_space(),
                    resources=resources_space,
                    ptr=pointer_space,
                )
            )
        )

    def build_dependencies(
        self, max_depth: int = None
    ) -> Generator[Tuple[Building, Optional[Building]], None, None]:
        buildings = [b for b in Buildings if not isinstance(b, Assimilator)]
        self.random.shuffle(buildings)
        n = len(buildings)
        if max_depth is not None:
            n = min(max_depth, n)
        dependencies = np.round(self.random.random(n) * np.arange(n)) - 1
        dependencies = [None if i < 0 else buildings[int(i)] for i in dependencies]

        yield Assimilator(), None
        yield from itertools.zip_longest(buildings, dependencies)

    def build_lines(self, dependencies: Dependencies) -> List[Line]:
        def instructions_for(building: Building):
            if building is None:
                return
            yield from instructions_for(dependencies[building])
            yield building

        def random_instructions_under(
            n: int, include_assimilator: bool = True
        ) -> Generator[List[Line], None, None]:
            if n < 0:
                raise RuntimeError
            if n == 0:
                return
            building, first, last, inst = None, None, None, None
            while None in (building, first, last, inst) or len(inst) > n:
                building = self.random.choice(
                    [
                        *filter(
                            lambda b: (
                                include_assimilator or not isinstance(b, Assimilator)
                            ),
                            Buildings,
                        )
                    ]
                )

                inst = *first, last = [*instructions_for(building)]
            for i in first:
                yield Line(False, i)
            yield Line(True, last)
            yield from random_instructions_under(
                n=n - len(inst),
                include_assimilator=include_assimilator
                and not isinstance(building, Assimilator),
            )

        n_lines = self.n_lines_space.sample()
        instructions = [*random_instructions_under(n_lines)]
        required = [i.building for i in instructions if i.required]
        assert required.count(Assimilator()) <= 1
        return instructions

    @staticmethod
    def build_trees(dependencies: Dependencies) -> typing.Set[Tree]:

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

    @staticmethod
    def done_generator():
        state: State
        state = yield

        while True:
            # noinspection PyTypeChecker
            state = yield state.success or not state.time_remaining, lambda: None

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
                if use_failure_buf or put_failure_buf:
                    i.update({"failure buffer size": self.failure_buffer.qsize()})

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
                    bucket_size = 5
                    lower = (len(lines) - 1) // bucket_size * bucket_size + 1
                    upper = (1 + (len(lines) - 1) // bucket_size) * bucket_size
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
                if len(lines) == 1 and elapsed_time > 0:
                    (line,) = lines
                    if line.building.cost.gas > 0:
                        info.update({"success on gas buildings": state.success})

            # noinspection PyTupleAssignmentBalance
            state, done = yield info, lambda: None
            info = {}
            elapsed_time += 1

    def main(self):
        keyboard_control.run(self, lambda: None)

    def obs_generator(self, *lines: Line):
        state: State
        state = yield

        padded: List[Optional[Line]] = [*lines, *[None] * (self.max_lines - len(lines))]
        line_mask = np.array([p is None for p in padded])

        def render():
            def lines_iterator():
                buildings = [*state.building_positions.values()]
                for l in lines:
                    built = l.building in buildings
                    yield Line(
                        required=not built,
                        building=l.building,
                    )
                    if built and l.required:
                        buildings.remove(l.building)

            for i, line in enumerate(list(lines_iterator())):
                print(
                    "{:2}{}{} {}".format(
                        i,
                        "-" if i == state.pointer else " ",
                        "*" if line.required else " ",
                        repr(line.building),
                    )
                )
            print("Obs:")
            for string in self.room_strings(array):
                print(string, end="")

        preprocessed = np.array([*map(self.preprocess_line, padded)])

        def coords():
            yield from state.positions.items()
            for p, b in state.building_positions.items():
                yield b, p

        while True:
            world = np.zeros((len(WorldObjects), *self.world_shape))
            for o, p in coords():
                world[(WorldObjects.index(o), *p)] = 1
            array = world
            resources = np.array([state.resources[r] for r in Resource])
            assert isinstance(state.action, CompoundAction)
            gate_openers = self.pad_gate_openers(np.array(state.action.gate_openers()))
            partial_action = np.array([*state.action.to_ints()])
            obs = OrderedDict(
                asdict(
                    Obs(
                        obs=array,
                        resources=resources,
                        line_mask=line_mask,
                        lines=preprocessed,
                        action_mask=state.action.mask().ravel(),
                        gate_openers=gate_openers,
                        partial_action=partial_action,
                        ptr=state.pointer,
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

    def pad_gate_openers(self, gate_openers):
        pad_amount, too_short = self.gate_opener_shape - np.array(gate_openers.shape)
        assert not too_short
        return np.pad(gate_openers, [(0, pad_amount), (0, 0)], mode="edge")

    def place_objects(self) -> Generator[Tuple[WorldObject, np.ndarray], None, None]:
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

        if self.num_initial_buildings:

            while True:
                initial_pos = self.random.choice(
                    self.world_size, size=(self.num_initial_buildings, 2)
                )
                initial_in_occupied = (
                    np.equal(
                        np.expand_dims(occupied, 0), np.expand_dims(initial_pos, 1)
                    )
                    .all(axis=-1)
                    .any()
                )
                if not initial_in_occupied:
                    initial_buildings = self.random.choice(
                        Buildings, size=self.num_initial_buildings
                    )
                    for b, p in zip(initial_buildings, initial_pos):
                        yield b, gas if isinstance(b, Assimilator) else p
                    return

        else:
            max_initial_buildings = max(
                0,
                (self.world_size ** 2 - len(occupied) - self.max_lines),
            )
            if max_initial_buildings > 0:
                num_initial_buildings = self.random.randint(max_initial_buildings + 1)
                initial_index = self.random.choice(
                    max_initial_buildings,
                    size=max_initial_buildings,
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
    def preprocess_line(line: Optional[Line]):
        if line is None:
            return [0, 0]
        return [int(line.required), Buildings.index(line.building)]

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    def reset(self):
        self.i += 1
        # self.iterator = self.failure_buffer_wrapper(self.srti_generator())
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
                (nonzero,) = channel.nonzero()
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
        dependencies = dict(self.build_dependencies())
        lines = self.build_lines(dependencies)
        obs_iterator = self.obs_generator(*lines)
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator(*lines)
        state_iterator = self.state_generator(lines, dependencies)
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        state, render_state = next(state_iterator)
        time_remaining = self.eval_steps

        def render():
            for tree in self.build_trees(dependencies):
                tree.show()

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
            if self.evaluating:
                time_remaining -= 1
                state = replace(state, time_remaining=time_remaining)

    def state_generator(
        self, lines: List[Line], dependencies: Dict[Building, Building]
    ) -> Generator[State, Optional[RawAction], None]:
        positions: List[Tuple[WorldObject, np.ndarray]] = [*self.place_objects()]
        building_positions: BuildingPositions = dict(
            [((i, j), b) for b, (i, j) in positions if isinstance(b, Building)]
        )
        pending_positions: BuildingPositions = {}
        positions: Positions = dict(
            [
                (o, (i, j))
                for o, (i, j) in positions
                if isinstance(o, (Resource, Worker))
            ]
        )
        assignments: Dict[Worker, Assignment] = {w: Resource.MINERALS for w in Worker}
        required = Counter(li.building for li in lines if li.required)
        resources: typing.Counter[Resource] = Counter()
        carrying: Carrying = {w: None for w in Worker}
        ptr: int = 0
        destroy = []
        action = NoWorkersAction()
        time_remaining = (1 + len(lines)) * self.time_per_line
        valid = True

        def render():
            print("Time remaining:", time_remaining)
            print("Resources:")
            pprint(resources)
            pprint(action if valid else new_action)
            for k, v in sorted(assignments.items()):
                print(f"{k}: {v}")
            if destroy:
                print(fg("red"), "Destroyed:", sep="")
                print(*destroy, sep="\n", end=RESET + "\n")
            if not valid:
                print(fg("red"), "Action not valid.", RESET, sep="")

        self.render_thunk = render

        while True:
            success = not required - Counter(building_positions.values())

            state = State(
                building_positions=building_positions,
                positions=positions,
                resources=resources,
                success=success,
                pointer=ptr,
                action=action,
                time_remaining=time_remaining,
            )

            a: Optional[RawAction]
            # noinspection PyTypeChecker
            a = yield state, render
            if a is None:
                a: List[ActionComponent] = [*action.from_input()]
            if isinstance(a, RawAction):
                a, ptr = a.a, a.ptr
            new_action = action.update(*a)
            valid = new_action.valid(
                resources=resources,
                dependencies=dependencies,
                building_positions=building_positions,
                pending_positions=pending_positions,
                positions=positions,
            )
            if not valid:
                time_remaining -= 1  # penalize agent for invalid
                continue

            action = new_action
            assignment = action.assignment(positions)
            is_op = assignment is not None
            if is_op:
                time_remaining -= 1
            else:
                continue

            for worker in action.get_workers():
                assignments[worker] = assignment

            worker_id: Worker
            for worker_id, assignment in sorted(
                assignments.items(),
                key=lambda w: isinstance(w[1], Resource),
                reverse=True,
            ):  # collect resources first.
                assignment.execute(
                    positions=positions,
                    worker=worker_id,
                    assignments=assignments,
                    building_positions=building_positions,
                    pending_positions=pending_positions,
                    resources=resources,
                    carrying=carrying,
                )

            destroy = []
            if self.random.random() < self.attack_prob:
                num_destroyed = self.random.randint(len(building_positions))
                destroy = [
                    c for c, b in building_positions.items() if not isinstance(b, Nexus)
                ]
                self.random.shuffle(destroy)
                destroy = destroy[:num_destroyed]
                for coord in destroy:
                    del building_positions[coord]

    def step(self, action: Union[np.ndarray, CompoundAction]):
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
        eval_steps=500,
        failure_buffer=failure_buffer,
    ).main()


if __name__ == "__main__":

    @dataclass
    class Config(EnvConfig):
        random_seed: int = 0

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    app()
