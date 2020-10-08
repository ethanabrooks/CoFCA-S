import copy
from collections import Counter, namedtuple, deque, OrderedDict
from itertools import product, zip_longest
from pprint import pprint
from typing import Tuple, Dict

import gym
import numpy as np
from colored import fg
from gym import spaces
from gym.utils import seeding

from lines import Subtask
from objects import (
    Terrain,
    Other,
    Refined,
    InventoryItems,
    WorldObjects,
    Necessary,
    Interaction,
    Resource,
    Symbols,
)
from utils import RESET

Coord = Tuple[int, int]
ObjectMap = Dict[Coord, str]
Last = namedtuple("Last", "action active reward terminal selected")
Action = namedtuple("Action", "upper lower delta dg ptr")
BuildBridge = Subtask(Interaction.BUILD, None)
Obs = namedtuple("Obs", "inventory inventory_change lines mask obs")
assert tuple(Obs._fields) == tuple(sorted(Obs._fields))


def delete_nth(d, n):
    d.rotate(-n)
    d.popleft()
    d.rotate(n)


def subtasks():
    yield BuildBridge
    for interaction in [Interaction.COLLECT, Interaction.REFINE]:
        for resource in Resource:
            yield Subtask(interaction, resource)


def lower_level_actions():
    yield Interaction.COLLECT
    for resource in Resource:
        yield resource  # REFINE
    for i in range(-1, 2):
        for j in range(-1, 2):
            if [i, j].count(0) == 1:
                yield np.array([i, j])


class Env(gym.Env):
    def __init__(
        self,
        bandit_prob: float,
        break_on_fail: bool,
        bridge_failure_prob: float,
        failure_buffer_size: int,
        map_discovery_prob: float,
        max_eval_lines: int,
        max_lines: int,
        min_eval_lines: int,
        min_lines: int,
        no_op_limit: int,
        rank: int,
        room_side: int,
        seed: int,
        tgt_success_rate: int,
        windfall_prob: float,
        evaluating=False,
    ):
        self.windfall_prob = windfall_prob
        self.bandit_prob = bandit_prob
        self.map_discovery_prob = map_discovery_prob
        self.bridge_failure_prob = bridge_failure_prob
        self.counts = None
        self.tgt_success_rate = tgt_success_rate

        self.subtasks = list(subtasks())
        self.blob_subtasks = [
            s for s in self.subtasks if s.interaction is not Interaction.BUILD
        ]
        num_subtasks = len(self.subtasks)
        self.min_eval_lines = min_eval_lines
        self.max_eval_lines = max_eval_lines
        self.rank = rank
        self.break_on_fail = break_on_fail
        self.no_op_limit = no_op_limit
        self.num_subtasks = num_subtasks
        self.i = 0
        self.success_count = 0

        self.min_lines = min_lines
        self.max_lines = max_lines
        if evaluating:
            self.n_lines = max_eval_lines
        else:
            self.n_lines = max_lines
        self.n_lines += 1
        self.random, self.seed = seeding.np_random(seed)
        self.failure_buffer = deque(maxlen=failure_buffer_size)
        self.non_failure_random = self.random.get_state()
        self.evaluating = evaluating
        self.iterator = None
        self.render_thunk = None
        self.h, self.w = self.room_shape = np.array([room_side, room_side])
        self.room_size = int(self.room_shape.prod())
        self.chunk_size = self.room_size - self.h - 1
        self.max_inventory = Counter({k: self.chunk_size for k in InventoryItems})
        self.limina = [Terrain.WATER] + [Terrain.WATER, Terrain.MOUNTAIN] * (self.h - 1)

        self.lower_level_actions = list(lower_level_actions())
        self.action_space = spaces.MultiDiscrete(
            np.array(
                Action(
                    upper=num_subtasks + 1,
                    delta=2 * self.n_lines,
                    dg=2,
                    lower=len(self.lower_level_actions),
                    ptr=self.n_lines,
                )
            )
        )
        self.line_space = [1 + len(Interaction), 2 + len(Resource)]
        lines_space = spaces.MultiDiscrete(np.array([self.line_space] * self.n_lines))
        mask_space = spaces.MultiDiscrete(2 * np.ones(self.n_lines))

        self.room_space = spaces.Box(
            low=np.zeros_like(self.room_shape),
            high=self.room_shape - 1,
            dtype=np.float32,
        )

        def inventory_space(n):
            return spaces.MultiDiscrete(
                n * np.ones(len(Resource) + len(Refined) + 1)  # +1 for map
            )

        self.observation_space = spaces.Dict(
            Obs(
                inventory=inventory_space(self.chunk_size),
                inventory_change=inventory_space(self.chunk_size + 1),
                lines=lines_space,
                mask=mask_space,
                obs=spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(Resource) + len(Terrain) + 1, *self.room_shape),
                    dtype=np.float32,
                ),
            )._asdict()
        )

    def reset(self):
        self.i += 1
        self.iterator = self.generator()
        s, r, t, i = next(self.iterator)
        return s

    def step(self, action: np.ndarray):
        action = Action(*action)
        action = action._replace(lower=self.lower_level_actions[action.lower])
        return self.iterator.send(action)

    @staticmethod
    def preprocess_line(line):
        if isinstance(line, Subtask):
            if line.resource is None:
                resource_code = len(Resource)
            else:
                resource_code = line.resource.value
            return [line.interaction.value, resource_code]
        elif line is None:
            return [0, 0]
        else:
            raise RuntimeError

    def room_strings(self, room):
        for i, row in enumerate(room.transpose((1, 2, 0)).astype(int)):
            for j, channel in enumerate(row):
                (nonzero,) = channel.nonzero()
                assert len(nonzero) <= 2
                for _, i in zip_longest(range(2), nonzero):
                    if i is None:
                        yield " "
                    else:
                        world_obj = WorldObjects[i]
                        yield Symbols[world_obj]
                yield RESET
                yield "|"
            yield "\n" + "-" * 3 * self.w + "\n"

    def preprocess_state(
        self,
        room,
        lines,
        inventory,
        info,
        done,
        ptr,
        subtask_complete,
        inventory_change,
        **kwargs
    ):
        _, padded = zip(*list(zip_longest(range(self.n_lines), lines)))
        obs = OrderedDict(
            Obs(
                obs=room,
                lines=[self.preprocess_line(l) for l in lines],
                mask=[p is None for p in padded],
                inventory=self.inventory_representation(inventory),
                inventory_change=self.inventory_representation(dict(inventory_change)),
            )._asdict()
        )
        for name, space in self.observation_space.spaces.items():
            if not space.contains(obs[name]):
                space.contains(obs[name])
        reward = -0.1
        return obs, reward, done, info

    def generator(self):
        use_failure_buf = self.use_failure_buf()
        if use_failure_buf:
            i = self.random.choice(len(self.failure_buffer))
            self.random.use_state(self.failure_buffer[i])
            delete_nth(self.failure_buffer, i)
        else:
            self.random.set_state(self.non_failure_random)
        state_iterator = self.state_generator()
        action = None
        while True:
            state = state_iterator.send(action)
            s, r, t, i = self.preprocess_state(**state)
            if t and not use_failure_buf:
                i.update(success_without_failure_buf=float(i["success"]))
            self.render_thunk = lambda: self._render(**state, reward=r, action=action)
            if t:
                self.non_failure_random = self.random.get_state()
            action = yield s, r, t, i

    def time_per_subtask(self):
        return 2 * (self.room_shape.sum() - 2)

    def state_generator(self):
        initial_random = self.random.get_state()
        blobs, rooms = self.initialize()
        assert len(rooms) == len(blobs)
        rooms_iter = iter(rooms)
        blobs_iter = iter(blobs)

        def get_lines():
            for blob in blobs:
                for subtask in blob:
                    yield subtask
                yield BuildBridge

        lines = list(get_lines())

        def next_required():
            blob = next(blobs_iter, None)
            if blob is not None:
                for subtask in blob:
                    if subtask.interaction == Interaction.COLLECT:
                        yield subtask.resource
                    elif subtask.interaction == Interaction.REFINE:
                        yield Refined(subtask.resource.value)
                    else:
                        assert subtask.interaction == Interaction.BUILD

        required = Counter(next_required())
        objects = dict(next(rooms_iter))
        agent_pos = int(self.random.choice(self.h)), int(self.random.choice(self.w - 1))
        inventory = Counter()
        rooms_complete = 0
        no_op_count = 0
        prev_inventory = inventory
        info = {}
        ptr = 0
        time_limit = self.n_lines * self.time_per_subtask()
        success = False
        subtask_complete = False
        done = False

        while True:
            time_limit -= 1
            fail = not self.evaluating and time_limit < 0
            self.success_count += int(success)

            done = done or fail or success
            if fail:
                self.failure_buffer.append(initial_random)
                if self.break_on_fail:
                    import ipdb

                    ipdb.set_trace()
            if done:
                info.update(instruction_len=len(lines), success=float(success))
                info.update(
                    rooms_complete=rooms_complete,
                    progress=rooms_complete / len(rooms),
                    success=float(success),
                )

            # line_specific_info = {
            #     f"{k}_{10 * (len(blobs) // 10)}": v for k, v in info.items()
            # }

            self.make_feasible(objects)
            room = self.obs_array(agent_pos, objects)
            inventory = inventory & self.max_inventory

            def inventory_change():
                for k in InventoryItems:
                    yield k, inventory[k] - prev_inventory[k]

            if subtask_complete:
                ptr += 1

            action = yield dict(
                room=room,
                lines=lines,
                inventory=inventory,
                inventory_change=inventory_change(),
                done=done,
                info=info,
                ptr=ptr,
                subtask_complete=subtask_complete,
                success=success,
                time_limit=time_limit,
                required=required,
            )
            prev_inventory = copy.deepcopy(inventory)
            subtask_complete = False

            info = dict(
                len_failure_buffer=len(self.failure_buffer),
            )

            no_op = action.upper == len(self.subtasks)
            if no_op:
                no_op_count += 1
                no_op_limit = 200 if self.evaluating else self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(blobs)
                if no_op_count >= no_op_limit:
                    done = True
                    continue

            if self.random.random() < self.bandit_prob:
                possessions = [k for k, v in inventory.items() if v > 0]
                if possessions:
                    robbed = self.random.choice(possessions)
                    inventory[robbed] -= 1

            line = lines[ptr]
            standing_on = objects.get(tuple(agent_pos), None)
            if isinstance(action.lower, np.ndarray):
                new_pos = agent_pos + action.lower
                moving_into = objects.get(tuple(new_pos), None)

                def next_room():
                    h, w = new_pos
                    return w == self.w

                def valid_new_pos():
                    if not self.room_space.contains(new_pos):
                        return next_room()
                    if moving_into == Terrain.WALL:
                        return False
                    if moving_into == Terrain.WATER:
                        return not (required - inventory)
                        # inventory dominates required
                    if moving_into == Terrain.MOUNTAIN:
                        return bool(inventory[Other.MAP])
                    return True

                if valid_new_pos():
                    if moving_into == Terrain.WATER:
                        inventory -= required  # build bridge
                        if self.random.random() > self.bridge_failure_prob:
                            agent_pos = new_pos  # check for "flash flood"
                        # else bridge failed
                    else:
                        agent_pos = new_pos % np.array(self.room_shape)
                        if moving_into == Terrain.MOUNTAIN:
                            inventory[Other.MAP] = 0
                        if next_room():
                            if line.interaction == Interaction.BUILD:
                                subtask_complete = True
                            room = next(rooms_iter, None)
                            if room is None:
                                success = True
                            else:
                                objects = dict(room)
                            required = Counter(next_required())
                            rooms_complete += 1
            elif action.lower == Interaction.COLLECT:
                if standing_on in list(Resource):
                    inventory[standing_on] += 1 + int(
                        self.random.random() < self.windfall_prob
                    )
                    del objects[tuple(agent_pos)]
                    if (
                        line.interaction == Interaction.COLLECT
                        and line.resource == standing_on
                    ):
                        subtask_complete = True
                    if self.random.random() < self.map_discovery_prob:
                        inventory[Other.MAP] = 1
            elif isinstance(action.lower, Resource):
                if standing_on == Terrain.FACTORY:
                    if (
                        line.interaction == Interaction.REFINE
                        and line.resource == action.lower
                    ):
                        subtask_complete = True
                    if inventory[action.lower]:
                        inventory[action.lower] -= 1
                        inventory[Refined(action.lower.value)] += 1
            else:
                raise NotImplementedError

    def use_failure_buf(self):
        if self.evaluating or len(self.failure_buffer) == 0:
            use_failure_buf = False
        else:
            success_rate = (1 + self.success_count) / self.i
            use_failure_prob = 1 - self.tgt_success_rate / success_rate
            use_failure_prob = max(use_failure_prob, 0)
            use_failure_buf = self.random.random() < use_failure_prob
        return use_failure_buf

    def obs_array(self, agent_pos, objects):
        room = np.zeros((len(WorldObjects), *self.room_shape))
        for p, o in list(objects.items()):
            p = np.array(p)
            room[(WorldObjects.index(o), *p)] = 1
        room[(WorldObjects.index(Other.AGENT), *agent_pos)] = 1
        return room

    def make_feasible(self, objects):
        counts = Counter(objects.values())
        if any(counts[o] == 0 for o in Necessary):

            def get_free_space():
                for i in range(self.h):
                    for j in range(self.w):
                        maybe_object = objects.get((i, j), None)
                        if maybe_object is None or (
                            maybe_object not in self.limina and counts[maybe_object] > 1
                        ):
                            yield i, j

            free_space = list(get_free_space())
            assert free_space

            for o in Necessary:
                if counts[o] == 0:
                    coord = free_space[self.random.choice(len(free_space))]
                    objects[coord] = o
                    counts = Counter(objects.values())
                    free_space = list(get_free_space())

    def initialize(self):
        n_lines = (
            self.random.random_integers(self.min_eval_lines, self.max_eval_lines)
            if self.evaluating
            else self.random.random_integers(self.min_lines, self.max_lines)
        )

        def get_blobs():
            i = n_lines
            while i > 0:
                size = self.random.choice(self.chunk_size)
                blob = self.random.choice(self.blob_subtasks, size=size)
                blob = blob[: i - 1]  # for BuildBridge
                yield blob
                i -= len(blob) + 1  # for BuildBridge

        blobs = list(get_blobs())
        n_rooms = len(blobs)
        assert n_rooms > 0
        h, w = self.room_shape
        coordinates = list(product(range(h), range(w - 1)))

        def get_objects():
            max_objects = len(coordinates)
            num_objects = self.random.choice(max_objects)
            h, w = self.room_shape
            positions = [
                coordinates[i]
                for i in self.random.choice(len(coordinates), size=num_objects)
            ] + [(i, w - 1) for i in range(h)]
            limina = copy.deepcopy(self.limina)
            self.random.shuffle(limina)
            limina = limina[: self.h]
            assert Terrain.WATER in limina
            world_objects = (
                list(self.random.choice(Necessary, size=num_objects)) + limina
            )
            assert len(positions) == len(world_objects)
            return list(zip(positions, world_objects))

        rooms = [get_objects() for _ in range(n_rooms)]
        return blobs, rooms

    def inventory_representation(self, inventory):
        return np.array([inventory[k] for k in InventoryItems])

    def seed(self, seed=None):
        assert self.seed == seed

    def _render(
        self,
        room,
        lines,
        inventory,
        done,
        ptr,
        success,
        time_limit,
        action,
        reward,
        required,
        **kwargs
    ):
        if done:
            print(fg("green") if success else fg("red"))

        for i, line in enumerate(lines):
            pre = "- " if i == ptr else "  "
            print("{:2}{}{}{}".format(i, pre, " ", str(line)))
        print(RESET)
        print("Time limit:", time_limit)
        print("Action:", action)
        print("Reward:", reward)
        print("Inventory:")
        pprint(inventory)
        print("Required:")
        pprint(required)
        print("Obs:")
        for string in self.room_strings(room):
            print(string, end="")

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    @classmethod
    def add_arguments(cls, p):
        p.add_argument("--min-lines", type=int)
        p.add_argument("--max-lines", type=int)
        p.add_argument("--no-op-limit", type=int)
        p.add_argument("--break-on-fail", action="store_true")
        p.add_argument("--tgt-success-rate", type=float)
        p.add_argument("--failure-buffer-size", type=int)
        p.add_argument("--room-side", type=int)
        p.add_argument("--bridge-failure-prob", type=float)
        p.add_argument("--map-discovery-prob", type=float)
        p.add_argument("--bandit-prob", type=float)
        p.add_argument("--windfall-prob", type=float)