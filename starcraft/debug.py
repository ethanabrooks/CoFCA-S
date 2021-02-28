import copy
import itertools
import sys
from collections import Counter, OrderedDict
from dataclasses import astuple, dataclass
from multiprocessing import Queue
from pprint import pprint
from typing import Generator, Tuple, Optional, List, Union

import gym
import hydra
import numpy as np
from colored import fg
from gym import spaces
from gym.spaces import MultiBinary, MultiDiscrete, Box
from gym.utils import seeding
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import keyboard_control
import osx_queue

from data_types import Obs, RawAction
from minecraft.data_types import Line
from starcraft.starcraft_data_types import (
    Building,
    Buildings,
    Unit,
    Units,
    UnitCounter,
    Resource,
    State,
    HighTemplar,
)
from starcraft.env import BuildingDependencies, UnitDependencies
from utils import RESET, Discrete, asdict


@dataclass
class EnvConfig:
    break_on_fail: bool = False
    max_lines: int = 10
    min_lines: int = 2
    no_ops: int = 3
    tgt_success_rate: float = 0.75
    check_spaces: bool = False


# noinspection PyAttributeOutsideInit
@dataclass
class Env(gym.Env):
    break_on_fail: bool
    check_spaces: bool
    failure_buffer: Queue
    max_lines: int
    min_lines: int
    no_ops: int
    rank: int
    random_seed: int
    tgt_success_rate: float
    alpha: float = 0.05
    evaluating: bool = None
    i: int = 0
    iterator = None
    render_thunk = None
    success_avg = 0.5
    success_with_failure_buf_avg = 0.5

    def __post_init__(self):
        self.random, _ = seeding.np_random(self.random_seed)
        self.n_lines_space = Discrete(self.min_lines, self.max_lines)
        self.n_lines_space.seed(self.random_seed)
        self.non_failure_random = self.random.get_state()

        self.act_spaces = RawAction(
            delta=2 * self.max_lines,
            gate=2,
            pointer=self.max_lines,
            extrinsic=Building.space().n + 1,  # +1 for no-op
        )
        self.action_space = spaces.MultiDiscrete(np.array(astuple(self.act_spaces)))

        # self.world_shape = world_shape = np.array([self.world_size, self.world_size])
        # self.world_space = spaces.Box(
        #     low=np.zeros_like(world_shape, dtype=np.float32),
        #     high=(world_shape - 1).astype(np.float32),
        # )

        # channel_size = len(WorldObjects) + 1  # +1 for destroy
        # max_shape = (channel_size, *world_shape)
        action_mask = spaces.MultiBinary(
            self.act_spaces.extrinsic
            #     action_components_space.nvec.max() * action_components_space.nvec.size
        )
        destroyed_unit = spaces.Discrete(Unit.space().n + 1)  # +1 for None
        gate_openers = spaces.MultiDiscrete(
            1
            + np.arange(self.act_spaces.extrinsic).reshape(-1, 1)
            # np.array(
            #     [CompoundAction.input_space().nvec] * ActionStage.gate_opener_max_size()
            # ).flatten()
        )
        instructions = spaces.MultiDiscrete(
            np.array(
                [Building.space().n + Unit.space().n + 1]  # +1 for padding
                * self.max_lines
            )
        )
        instruction_mask = spaces.MultiDiscrete(2 * np.ones(self.max_lines))
        obs = spaces.Box(low=0, high=1, shape=(Building.space().n, 1, 1))
        # obs = spaces.Box(
        #     low=np.zeros(max_shape, dtype=np.float32),
        #     high=np.ones(max_shape, dtype=np.float32),
        # )
        # partial_action = CompoundAction.representation_space()
        partial_action = spaces.MultiDiscrete([self.act_spaces.extrinsic])
        resources = spaces.MultiDiscrete([sys.maxsize] * 2)
        pointer = spaces.Discrete(self.max_lines)
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
            pointer=pointer,
        )

        self.observation_space = gym.spaces.Dict(asdict(self.obs_spaces))
        self.action_space = spaces.MultiDiscrete(np.array(astuple(self.act_spaces)))

    @staticmethod
    def done_generator():
        state: State
        state = yield
        while True:
            state: State
            term = (
                not state.time_remaining or not state.no_ops_remaining or state.success
            )
            # noinspection PyTypeChecker
            state = (yield term, lambda: None)

    @staticmethod
    def info_generator():
        state = None
        i = {}
        term = False
        while True:
            # noinspection PyTupleAssignmentBalance
            if term:
                assert state is not None
                i = dict(success=state.success)
            state: Optional[State]
            (state, term) = yield i, lambda: None

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
                if building is not None and not state.buildings[building]:
                    # if building not in [None, *state.buildings.values()]:
                    yield building
                    yield from buildings_required_for(building)

            def get_required_buildings():
                for unit in state.required_units:
                    yield from buildings_required_for(unit)

            required_buildings = set(get_required_buildings())

            for i, line in enumerate(lines):
                if i == state.agent_pointer:
                    prefix = "-"
                else:
                    prefix = " "
                print(
                    "{:2}{}{} ({}) {}".format(
                        i,
                        prefix,
                        "*"
                        if line in [*required_buildings, *state.required_units]
                        else " ",
                        [*Buildings, *Units].index(line),
                        line,
                    )
                )
            print("Obs:")
            pprint(state.buildings)
            # for string in self.room_strings(world):
            #     print(string, end="")

        # def coords():
        #     yield from state.positions.items()
        #     for p, b in state.buildings.items():
        #         yield b, p

        while True:
            # world = np.zeros(self.obs_spaces.obs.shape)
            # for o, p in coords():
            #     world[(WorldObjects.index(o), *p)] = 1
            # for p in state.destroyed_buildings.keys():
            #     world[(-1, *p)] = 1
            # assert isinstance(state.action, ActionStage)

            # gate_openers: np.ndarray = self.obs_spaces.gate_openers.nvec.copy().reshape(
            #     -1, CompoundAction.input_space().nvec.size
            # )
            # gate_openers -= 1
            # unpadded_gate_openers = state.action.gate_openers()
            # gate_openers[: len(unpadded_gate_openers)] = unpadded_gate_openers
            world = np.array([bool(state.buildings[b]) for b in Buildings]).reshape(
                (-1, 1, 1)
            )

            if state.destroyed_unit is None:
                destroyed_unit = 0
            else:
                destroyed_unit = 1 + state.destroyed_unit.to_int()
            num_actions = self.act_spaces.extrinsic
            action_mask = np.zeros(self.act_spaces.extrinsic)
            obs = OrderedDict(
                asdict(
                    Obs(
                        action_mask=action_mask,
                        destroyed_unit=destroyed_unit,
                        gate_openers=np.arange(num_actions).reshape(-1, 1),
                        # gate_openers=gate_openers.ravel(),
                        instruction_mask=np.array([int(p is None) for p in padded]),
                        instructions=np.array([*map(self.preprocess_line, padded)]),
                        obs=world,
                        partial_action=np.array([0]),
                        pointer=state.agent_pointer,
                        resources=(np.array([state.resources[r] for r in Resource])),
                    )
                )
            )
            if self.check_spaces:
                for (k, space), (n, o) in zip(
                    self.observation_space.spaces.items(), obs.items()
                ):
                    if not space.contains(o):
                        import ipdb

                        ipdb.set_trace()
                        space.contains(o)
            # noinspection PyTypeChecker
            state = yield obs, lambda: render()  # perform time-step

    @staticmethod
    def reward_generator():
        state: State
        state = yield

        while True:
            reward = float(state.success)
            # noinspection PyTypeChecker
            state = yield reward, lambda: print("Reward:", reward)

    def state_generator(
        self, *instructions: Line
    ) -> Generator[bool, Optional[RawAction], None]:
        action = None
        building = None
        success = False

        required_units: UnitCounter = Counter(
            l for l in instructions if isinstance(l, Unit)
        )
        destroyed_index = self.random.choice(
            [i for i, l in enumerate(instructions) if isinstance(l, Unit)]
        )
        destroyed_unit = instructions[destroyed_index]

        time_remaining = 1
        no_ops_remaining = self.no_ops

        def render():
            print(instructions)
            print()
            print("Destroyed:", destroyed_unit)
            print("Action:", action)
            print("Building:", building)
            print()

        self.render_thunk = render

        def first_dependency() -> Building:
            prev = destroyed_unit
            for line in reversed(instructions[:destroyed_index]):
                if isinstance(line, Unit):
                    assert isinstance(prev, Building)
                    return prev
                prev = line
            return line

        while True:
            state = State(
                agent_pointer=destroyed_index if action is None else action.pointer,
                success=success,
                buildings=Counter(),
                required_units=required_units,
                resources=Counter(),
                destroyed_unit=destroyed_unit,
                time_remaining=time_remaining,
                no_ops_remaining=no_ops_remaining,
            )
            # noinspection PyTypeChecker
            action = (yield state, render)
            if action.extrinsic == 0:
                no_ops_remaining -= 1
            else:
                time_remaining -= 1
                building = Building.parse(int(action.extrinsic - 1))
                first = first_dependency()
                success = building == first

    def srti_generator(
        self,
    ) -> Generator[Tuple[any, float, bool, dict], Optional[RawAction], None]:
        instructions, deps, unit_deps = self.build_instructions_and_dependencies()
        obs_iterator = self.obs_generator(
            *instructions, building_dependencies=deps, unit_dependencies=unit_deps
        )
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator()
        state_iterator = self.state_generator(*instructions)
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        state, render_state = next(state_iterator)

        def render():
            if t:
                print("srti", i)
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
            if not isinstance(a, RawAction):
                a = RawAction.parse(*a)

            state, render_state = state_iterator.send(a)

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

        # yield Assimilator(), None
        yield from itertools.zip_longest(buildings, dependencies)

    def build_unit_dependencies(
        self,
    ) -> Generator[Tuple[Unit, Building], None, None]:
        buildings = self.random.choice(Buildings, size=len(Units))
        units = copy.copy(Units)
        self.random.shuffle(units)
        yield from zip(units, buildings)

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
            unit = self.random.choice(possible_units)
            building = unit_dependencies[unit]
            _instructions = [*_instructions_for[building], unit]
            yield from _instructions
            yield from random_instructions(n - len(_instructions))

        instructions = [*random_instructions(n_lines)]
        if not instructions:
            return self.build_instructions_and_dependencies()
        assert len(instructions) >= 2
        return instructions, building_dependencies, unit_dependencies

    def main(self):
        def action_fn():
            while True:
                action = input("go:")
                try:
                    action = int(action)
                    return RawAction.parse(0, 0, 0, action)
                except ValueError:
                    pass

        keyboard_control.run(self, action_fn)

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

    def seed(self, seed=None):
        assert self.random_seed == seed

    def step(self, action: np.ndarray):
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
