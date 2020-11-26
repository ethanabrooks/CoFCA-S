import pickle
import typing
from collections import Counter, deque, OrderedDict
from dataclasses import astuple, asdict, dataclass, replace
from functools import reduce
from itertools import zip_longest
from pathlib import Path
from pprint import pprint
from typing import Union, Dict, Generator, Tuple, List, Optional

import gym
import numpy as np
from colored import fg
from gym import spaces
from gym.spaces import MultiDiscrete
from gym.utils import seeding
from treelib import Tree

import keyboard_control
from data_types import (
    Obs,
    Resource,
    Building,
    Coord,
    Costs,
    costs,
    WorldObject,
    WorldObjects,
    Movement,
    Worker,
    Resources,
    State,
    Line,
    Symbols,
    BuildOrder,
    CompoundAction,
    RawAction,
    Assignment,
    Action1,
    Action2,
    Targets,
    Action3,
    WorkerAction,
    WorkerActions,
    WORLD_SIZE,
)

from utils import RESET

Dependencies = Dict[Building, Building]


def delete_nth(d, n):
    d.rotate(-n)
    d.popleft()
    d.rotate(n)


@dataclass
class Env(gym.Env):
    break_on_fail: bool
    destroy_building_prob: float
    eval_steps: int
    failure_buffer_load_path: Path
    failure_buffer_size: int
    max_lines: int
    min_lines: int
    num_initial_buildings: int
    rank: int
    random_seed: int
    tgt_success_rate: int
    alpha: float = 0.05
    evaluating: bool = None
    i: int = 0
    iterator = None
    render_thunk = None
    success_avg = 0.5

    def __post_init__(self):
        super().__init__()
        self.world_size = WORLD_SIZE
        self.random, _ = seeding.np_random(self.random_seed)
        self.failure_buffer = deque(maxlen=self.failure_buffer_size)
        if self.failure_buffer_load_path:
            with self.failure_buffer_load_path.open("rb") as f:
                self.failure_buffer.extend(pickle.load(f))
                print(
                    f"Loaded failure buffer of length {len(self.failure_buffer)} "
                    f"from {self.failure_buffer_load_path}"
                )
        self.non_failure_random = self.random.get_state()
        self.a_size = max_a_action = max(
            [c.size_a() for c in self.compound_action().classes()]
        )
        self.action_space = spaces.MultiDiscrete(
            np.array(
                astuple(
                    RawAction(
                        delta=2 * self.max_lines,
                        dg=2,
                        a=max_a_action,
                        ptr=self.max_lines,
                    )
                )
            )
        )

        lines_space = spaces.MultiDiscrete(
            np.array([[2, len(Building)]] * self.max_lines)
        )
        mask_space = spaces.MultiDiscrete(2 * np.ones(self.max_lines))
        self.world_shape = world_shape = np.array([self.world_size, self.world_size])
        self.world_space = spaces.Box(
            low=np.zeros_like(world_shape, dtype=np.float32),
            high=(world_shape - 1).astype(np.float32),
        )

        shape = (len(WorldObjects), *world_shape)
        obs_space = spaces.Box(
            low=np.zeros(shape, dtype=np.float32),
            high=np.ones(shape, dtype=np.float32),
        )
        self.max = Resources(*sum(Costs.values(), Counter()).values())
        self.time_per_line = 2 * max(
            reduce(lambda a, b: a | b, Costs.values(), Costs[Building.NEXUS]).values()
        )
        resources_space = spaces.MultiDiscrete(
            1 + np.array([self.max.minerals, self.max.gas])
        )
        next_actions_space = MultiDiscrete(np.ones(len(Worker)) * len(WorkerActions))
        partial_action_space = spaces.MultiDiscrete(
            [
                1 + a
                for c in self.compound_action().classes()
                for a in astuple(c.num_values())
            ]  # [:-1]
        )

        # noinspection PyTypeChecker
        self.observation_space = spaces.Dict(
            asdict(
                Obs(
                    action_mask=spaces.MultiBinary(max_a_action),
                    can_open_gate=spaces.MultiBinary(max_a_action),
                    lines=lines_space,
                    mask=mask_space,
                    next_actions=next_actions_space,
                    obs=obs_space,
                    partial_action=partial_action_space,
                    resources=resources_space,
                )
            )
        )

    @classmethod
    def add_arguments(cls, p):
        p.add_argument("--break-on-fail", action="store_true")
        p.add_argument("--debug-env", action="store_true")
        p.add_argument("--destroy-building-prob", type=float)
        p.add_argument("--failure-buffer-load-path", type=Path, default=None)
        p.add_argument("--failure-buffer-size", type=int)
        p.add_argument("--min-lines", type=int)
        p.add_argument("--max-lines", type=int)
        p.add_argument("--num-initial-buildings", type=int)
        p.add_argument("--tgt-success-rate", type=float)

    def build_dependencies(self):
        n = len(Building)
        dependencies = np.round(self.random.random(n) * np.arange(n)).astype(int) - 1
        buildings = list(Building)
        self.random.shuffle(buildings)

        def generate_dependencies():
            for b1, b2 in zip(buildings, dependencies):
                yield b1, (
                    None if b1 is Building.ASSIMILATOR or b2 < 0 else buildings[b2]
                )

        return dict(generate_dependencies())

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
                            lambda b: b is not Building.NEXUS
                            and (include_assimilator or b is not Building.ASSIMILATOR),
                            Building,
                        )
                    ]
                )

                inst = *first, last = [*instructions_for(building)]
                assert last is not Building.NEXUS
            for i in first:
                yield Line(False, i)
            yield Line(True, last)
            yield from random_instructions_under(
                n=n - len(inst),
                include_assimilator=include_assimilator
                and building is not Building.ASSIMILATOR,
            )

        n_lines = self.random.randint(self.min_lines, self.max_lines + 1)
        instructions = [*random_instructions_under(n_lines)]
        required = [l.building for l in instructions if l.required]
        assert required.count(Building.ASSIMILATOR) <= 1
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
            trees[bldg].create_node(bldg.name.capitalize(), bldg, parent=dependency)

        for building in Building:
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
        if self.evaluating or len(self.failure_buffer) == 0:
            buf = False
        else:
            use_failure_prob = 1 - self.tgt_success_rate / self.success_avg
            use_failure_prob = max(use_failure_prob, 0)
            buf = self.random.random() < use_failure_prob
        use_failure_buf = buf
        if use_failure_buf:
            i = self.random.choice(len(self.failure_buffer))
            self.random.set_state(self.failure_buffer[i])
            delete_nth(self.failure_buffer, i)
        else:
            self.random.set_state(self.non_failure_random)
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
            if not use_failure_buf:
                i.update(reward_without_failure_buf=r)
            if t:
                success = i["success"]
                if not use_failure_buf:
                    i.update(success_without_failure_buf=float(success))
                    self.success_avg += self.alpha * (success - self.success_avg)

                i.update(
                    use_failure_buf=use_failure_buf,
                )
                if not self.evaluating:
                    i.update(failure_buffer=[*self.failure_buffer][:2])

                if not success:
                    self.failure_buffer.append(initial_random)

            if t:
                # noinspection PyAttributeOutsideInit
                self.non_failure_random = self.random.get_state()
            action = yield s, r, t, i

    def info_generator(self, *lines):
        state: State
        done: bool
        state, done = yield
        info = dict(len_failure_buffer=float(len(self.failure_buffer)))

        while True:
            if done:
                info.update(
                    instruction_len=len(lines),
                    len_failure_buffer=len(self.failure_buffer),
                    success=float(state.success),
                    train_time_success=float(state.success and state.time_remaining),
                )
                if self.evaluating:
                    bucket = 10 * (len(lines) // 10)
                    for key in [
                        "success",
                        "train_time_success",
                        "normalized_elapsed_time",
                    ]:
                        info[f"{key}{bucket}"] = info[key]
            state, done = yield info, lambda: None
            info = {}

    def main(self):
        def action_fn(string: str):
            try:
                *args, target = map(int, string.split())
                try:
                    i, j = args
                    worker = 0
                except ValueError:
                    if Targets[target] is Building:
                        print("Must specify i/j for buildings.")
                        return
                    i, j = 0, 0
                    worker = 1
                worker = Worker(worker + 1)
                target = Targets[target]
                return self.compound_action(
                    Action1(is_op=True),
                    Action2(worker, target),
                    Action3(i, j),
                )
            except (ValueError, TypeError) as e:
                print(e)

        keyboard_control.run(self, action_fn)

    def obs_generator(self, *lines: Line):
        state: State
        state = yield
        padded: List[Optional[Line]] = [*lines, *[None] * (self.max_lines - len(lines))]
        mask = np.array([p is not None for p in padded])

        def render():
            def lines_iterator():
                buildings = [*state.building_positions.values()]
                for l in lines:
                    built = l.building in buildings
                    yield Line(
                        required=l.required and not built,
                        building=l.building,
                    )
                    if built and l.required:
                        buildings.remove(l.building)

            for i, line in enumerate(list(lines_iterator())):
                print(
                    "{:2}{}{} ({}) {}: {}".format(
                        i,
                        "-" if i == state.pointer else " ",
                        "*" if line.required else " ",
                        Targets.index(line.building),
                        str(line.building),
                        costs[line.building],
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
            next_actions = np.array(
                [WorkerActions.index(a) for a in state.next_action.values()]
            )
            action_mask = np.array([*state.action.mask(self.a_size)])
            can_open_gate = np.array([*state.action.can_open_gate(self.a_size)])
            partial_action = np.array([*state.action.partial_actions()])
            obs = OrderedDict(
                asdict(
                    Obs(
                        obs=array,
                        resources=resources,
                        lines=preprocessed,
                        mask=mask,
                        next_actions=next_actions,
                        action_mask=action_mask,
                        can_open_gate=can_open_gate,
                        partial_action=partial_action,
                    )
                )
            )
            # for (k, space), (n, o) in zip(
            # self.observation_space.spaces.items(), obs.items()
            # ):
            # if not space.contains(o):
            # import ipdb

            # ipdb.set_trace()
            # space.contains(o)
            # noinspection PyTypeChecker
            state = yield obs, lambda: render()  # perform time-step

    def place_objects(self) -> Generator[Tuple[WorldObject, np.ndarray], None, None]:
        nexus = self.random.choice(self.world_size, size=2)
        yield Building.NEXUS, nexus
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
        while True:
            initial_pos = self.random.choice(
                self.world_size, size=(self.num_initial_buildings, 2)
            )
            initial_in_occupied = (
                np.equal(np.expand_dims(occupied, 0), np.expand_dims(initial_pos, 1))
                .all(axis=-1)
                .any()
            )
            if not initial_in_occupied:
                initial_buildings = self.random.choice(
                    Building, size=self.num_initial_buildings
                )
                for b, p in zip(initial_buildings, initial_pos):
                    yield b, gas if b is Building.ASSIMILATOR else p
                return

    @staticmethod
    def preprocess_line(line: Optional[Line]):
        if line is None:
            return [0, 0]
        return [int(line.required), line.building.value - 1]

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    def reset(self):
        self.i += 1
        self.iterator = self.failure_buffer_wrapper(self.srti_generator())
        s, r, t, i = next(self.iterator)
        return s

    def room_strings(self, room):
        grid_size = 5
        for i, row in enumerate(room.transpose((1, 2, 0)).astype(int)):
            for j, channel in enumerate(row):
                (nonzero,) = channel.nonzero()
                assert len(nonzero) <= grid_size
                for _, i in zip_longest(range(grid_size), nonzero):
                    if i is None:
                        yield " "
                    else:
                        world_obj = WorldObjects[i]
                        yield Symbols[world_obj]
                yield RESET
                yield "|"
            yield "\n" + "-" * (grid_size + 1) * self.world_size + "\n"

    @staticmethod
    def reward_generator():
        reward = -0.1
        while True:
            yield reward, lambda: print("Reward:", reward)

    def seed(self, seed=None):
        assert self.random_seed == seed

    def srti_generator(
        self,
    ) -> Generator[Tuple[any, float, bool, dict], RawAction, None]:
        dependencies = self.build_dependencies()
        lines = self.build_lines(dependencies)
        obs_iterator = self.obs_generator(*lines)
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator(*lines)
        state_iterator = self.state_generator(*lines)
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        action = self.compound_action()
        state, render_state = next(state_iterator)

        def render():
            for tree in self.build_trees(dependencies):
                tree.show()

            if t:
                print(fg("green") if i["success"] else fg("red"))
            render_r()
            render_t()
            render_i()
            render_state()
            print("Action:", end=" ")
            print(action)
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

            # noinspection PyTypeChecker
            a = yield s, r, t, i
            action = action.update(a)
            state = replace(state, action=action)

            if action.is_op():
                state, render_state = state_iterator.send(action)
            elif self.evaluating:
                state = replace(state, time_remaining=state.time_remaining - 1)

    @staticmethod
    def compound_action(*args, **kwargs) -> CompoundAction:
        return CompoundAction(*args, **kwargs)

    @staticmethod
    def compound_action(*args, **kwargs) -> CompoundAction:
        return CompoundAction(*args, **kwargs)

    def state_generator(self, *lines: Line) -> Generator[State, CompoundAction, None]:
        positions: List[Tuple[WorldObject, np.ndarray]] = [*self.place_objects()]
        building_positions: Dict[Coord, Building] = dict(
            [((i, j), b) for b, (i, j) in positions if isinstance(b, Building)]
        )
        positions: Dict[Union[Resource, Worker], Coord] = dict(
            [(o, (i, j)) for o, (i, j) in positions if not isinstance(o, Building)]
        )
        assignments: Dict[Worker, Assignment] = {}
        next_actions: Dict[Worker, WorkerAction] = {}
        for worker_id in Worker:
            assignments[worker_id] = self.initial_assignment()

        required = Counter(l.building for l in lines if l.required)
        resources: typing.Counter[Resource] = Counter()
        ptr: int = 0
        action = self.compound_action()
        time_remaining = (
            self.eval_steps - 1 if self.evaluating else len(lines) * self.time_per_line
        )

        while True:
            destroyed_buildings = [
                (c, b)
                for c, b in building_positions.items()
                if self.random.random() < self.destroy_building_prob
                and b is not Building.NEXUS
            ]
            if destroyed_buildings:
                destroy_coords, destroyed_buildings = zip(*destroyed_buildings)
                for coord in destroy_coords:
                    del building_positions[coord]

            success = not required - Counter(building_positions.values())

            state = State(
                building_positions=building_positions,
                next_action=next_actions,
                positions=positions,
                resources=resources,
                success=success,
                pointer=ptr,
                action=action,
                time_remaining=time_remaining,
            )

            def render():
                print("Time remaining:", time_remaining)
                print("Resources:")
                pprint(resources)
                pprint(assignments)
                if destroyed_buildings:
                    print(fg("red"), "Destroyed:", sep="")
                    print(*destroyed_buildings, sep="\n", end=RESET + "\n")

            self.render_thunk = render

            nexus_positions: List[Coord] = [
                p for p, b in building_positions.items() if b is Building.NEXUS
            ]
            assert nexus_positions
            for worker_id, assignment in assignments.items():
                next_actions[worker_id] = assignment.action(
                    positions[worker_id],
                    positions,
                    [p for p, b in building_positions.items() if b is Building.NEXUS],
                )

            action: CompoundAction
            # noinspection PyTypeChecker
            action = yield state, render
            ptr = action.ptr
            time_remaining -= 1
            assignments[action.worker()] = action.assignment()

            worker_id: Worker
            assignment: Assignment
            for worker_id, assignment in sorted(
                assignments.items(), key=lambda w: isinstance(w[1], BuildOrder)
            ):  # collect resources first.
                worker_position = positions[worker_id]
                worker_action = assignment.action(
                    current_position=worker_position,
                    positions=positions,
                    nexus_positions=nexus_positions,
                )

                if isinstance(worker_action, Movement):
                    new_position = tuple(
                        np.array(worker_position) + np.array(astuple(worker_action))
                    )
                    positions[worker_id] = new_position
                    if building_positions.get(new_position, None) == Building.NEXUS:
                        for resource in Resource:
                            if self.gathered_resource(
                                building_positions, positions, resource, worker_position
                            ):
                                resources[resource] += 1
                elif isinstance(worker_action, Building):
                    building = worker_action
                    insufficient_resources = Costs[building] - resources
                    if self.building_allowed(
                        building,
                        building_positions,
                        insufficient_resources,
                        positions,
                        assignment.location,
                    ):
                        building_positions[worker_position] = building
                        resources -= Costs[building]
                else:
                    raise RuntimeError

    def gathered_resource(
        self, building_positions, positions, resource, worker_position
    ):
        return positions[resource] == worker_position and (
            resource != Resource.GAS
            or building_positions.get(worker_position, None) == Building.ASSIMILATOR
        )

    @staticmethod
    def initial_assignment():
        return Resource.MINERALS

    @staticmethod
    def building_allowed(
        building,
        building_positions,
        insufficient_resources,
        positions,
        assignment_location,
    ) -> bool:
        if insufficient_resources or assignment_location in building_positions:
            return False
        if building is Building.ASSIMILATOR:
            return assignment_location == positions[Resource.GAS]
        else:
            return assignment_location not in (
                *building_positions,
                positions[Resource.GAS],
                positions[Resource.MINERALS],
            )

    def step(self, action: Union[np.ndarray, CompoundAction]):
        if isinstance(action, np.ndarray):
            action = RawAction(*action)
        return self.iterator.send(action)

    def time_limit(self, lines):
        return len(lines) * self.time_per_line


def main(debug_env: bool, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
