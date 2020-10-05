from collections import Counter, namedtuple, deque, OrderedDict
from enum import unique, Enum, auto
from itertools import product, zip_longest
from pprint import pprint
from typing import Tuple, Dict, Any

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import keyboard_control
from lines import Subtask, Interaction, Resource
from utils import (
    hierarchical_parse_args,
    RESET,
    GREEN,
    RED,
    LIGHTGREY,
    PINK,
    BLUE,
    DARKGREY,
)

Coord = Tuple[int, int]
ObjectMap = Dict[Coord, str]

Obs = namedtuple("Obs", "inventory inventory_change lines mask obs")
assert tuple(Obs._fields) == tuple(sorted(Obs._fields))

Last = namedtuple("Last", "action active reward terminal selected")
Action = namedtuple("Action", "upper lower delta dg ptr")

BuildBridge = Interaction.BUILD, None

"""
TODO:
add to failure_buffer
"""


@unique
class Terrain(Enum):
    FACTORY = auto()
    WATER = auto()
    MOUNTAIN = auto()
    WALL = auto()


@unique
class Other(Enum):
    AGENT = auto()
    MAP = auto()


Colors = {
    Resource.WOOD: GREEN,
    Resource.STONE: LIGHTGREY,
    Resource.IRON: DARKGREY,
    Terrain.FACTORY: PINK,
    Terrain.WALL: RESET,
    Terrain.WATER: BLUE,
}

Refined = Enum(value="Refined", names=[x.name for x in Resource])
InventoryItems = list(Resource) + list(Refined) + [Other.MAP]
WorldObjects = list(Resource) + list(Refined) + [Other.AGENT]
Necessary = list(Resource) + [Terrain.FACTORY]


def subtasks():
    yield BuildBridge
    for interaction in Interaction:
        for resource in Resource:
            yield interaction, resource


class Env(gym.Env):
    def __init__(
        self,
        max_failure_sample_prob: int,
        failure_buffer_size: int,
        min_eval_lines: int,
        max_eval_lines: int,
        min_lines: int,
        max_lines: int,
        no_op_limit: int,
        time_to_waste: int,
        subtasks_only: bool,
        break_on_fail: bool,
        rank: int,
        lower_level: str,
        bridge_failure_prob=0.25,
        map_discovery_prob=0.02,
        bandit_prob=0.005,
        windfall_prob=0.25,
        evaluating=False,
        max_nesting_depth=1,
        seed=0,
        room_shape=(3, 3),
    ):
        self.windfall_prob = windfall_prob
        self.bandit_prob = bandit_prob
        self.map_discovery_prob = map_discovery_prob
        self.bridge_failure_prob = bridge_failure_prob
        self.counts = None
        self.max_failure_sample_prob = max_failure_sample_prob
        self.failure_buffer = deque(maxlen=failure_buffer_size)

        self.subtasks = list(subtasks())
        num_subtasks = len(self.subtasks)
        self.min_eval_lines = min_eval_lines
        self.max_eval_lines = max_eval_lines
        self.lower_level = lower_level
        self.rank = rank
        self.break_on_fail = break_on_fail
        self.subtasks_only = subtasks_only
        self.no_op_limit = no_op_limit
        self.max_nesting_depth = max_nesting_depth
        self.num_subtasks = num_subtasks
        self.time_to_waste = time_to_waste
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
        self.evaluating = evaluating
        self.iterator = None
        self._render = None
        self.h, self.w = self.room_shape = np.array(room_shape)
        self.room_size = int(self.room_shape.prod())
        self.chunk_size = self.max_inventory = self.room_size - 1
        self.limina = [Terrain.WATER] + [Terrain.WATER, Terrain.MOUNTAIN] * self.h

        def lower_level_actions():
            yield Interaction.COLLECT
            for resource in Resource:
                yield Interaction.REFINE, resource
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 not in (i, j):
                        yield np.array([i, j])

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
        lines_space = spaces.MultiDiscrete(
            np.array([[len(Interaction), len(Resource)]] * self.n_lines)
        )
        mask_space = spaces.MultiDiscrete(2 * np.ones(self.n_lines))

        self.room_space = spaces.Box(
            low=np.zeros_like(self.room_shape),
            high=self.room_shape,
            dtype=np.float32,
        )

        def inventory_space(n):
            return spaces.MultiDiscrete(n * np.ones(len(Resource)))

        self.observation_space = spaces.Dict(
            Obs(
                inventory=inventory_space(self.max_inventory),
                inventory_change=inventory_space(self.max_inventory + 1),
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

    def step(self, action):
        return self.iterator.send(action)

    def generator(self):
        use_failure_buf = (
            not self.evaluating
            and len(self.failure_buffer) > 0
            and (
                self.random.random()
                < self.max_failure_sample_prob * self.success_count / self.i
            )
        )
        if use_failure_buf:
            choice = self.random.choice(len(self.failure_buffer))
            blobs, rooms = self.failure_buffer.pop(choice)
        else:
            n_lines = (
                self.random.random_integers(self.min_eval_lines, self.max_eval_lines)
                if self.evaluating
                else self.random.random_integers(self.min_lines, self.max_lines)
            )

            def get_blob():
                for i in self.random.choice(
                    len(self.subtasks), size=self.random.choice(self.chunk_size)
                ):
                    yield Subtask(*self.subtasks[i])

            def get_blobs():
                i = n_lines - 1  # for BuildBridge
                while i > 0:
                    chunk = list(get_blob())[:i]
                    yield chunk
                    i -= len(chunk) + 1  # for BuildBridge

            blobs = list(get_blobs())
            n_rooms = len(blobs)
            assert n_rooms > 0
            h, w = self.room_shape
            coordinates = list(product(range(h), range(w)))

            def get_objects():
                max_objects = len(coordinates)
                num_objects = self.random.choice(max_objects)
                h, w = self.room_shape
                positions = [
                    coordinates[i]
                    for i in self.random.choice(len(coordinates), size=num_objects)
                ] + [(i, w - 1) for i in range(h)]
                self.random.shuffle(self.limina)
                limina = self.limina[: self.h]
                assert Terrain.WATER in limina
                world_objects = (
                    list(self.random.choice(WorldObjects, size=num_objects)) + limina
                )
                assert len(positions) == len(world_objects)
                return list(zip(positions, world_objects))

            rooms = [get_objects() for _ in range(n_rooms)]

        assert len(rooms) == len(blobs)
        rooms_iter = iter(rooms)
        blobs_iter = iter(blobs)

        def next_required():
            blob = next(blobs_iter, None)
            if blob is not None:
                return Counter([subtask.resource for subtask in blob])

        def no_op(action_index):
            return action_index == len(self.subtasks)

        objects = next(rooms_iter)
        agent_pos, _ = objects.pop()  # type: Tuple[Coord, Any]
        objects = dict(objects)
        inventory = Counter()
        inventory_change = Counter()
        rooms_complete = 0
        no_op_count = 0
        required = list(next_required())
        fail = False
        info = {}
        ptr = 0

        def get_lines():
            for blob in blobs:
                for subtask in blob:
                    yield subtask
                yield BuildBridge

        _, padded = zip(*list(zip_longest(range(self.n_lines), get_lines())))
        import ipdb

        ipdb.set_trace()

        def preprocess_lines():
            for _, line in padded:
                if isinstance(line, Subtask):
                    if line.resource is None:
                        resource_code = len(Resource)
                    else:
                        resource_code = line.resource.value
                    yield [
                        1 + line.interaction.value,
                        1 + resource_code,
                    ]
                elif line is None:
                    yield [0, 0]
                else:
                    raise RuntimeError

        while True:
            success = objects is None
            self.success_count += success

            done = fail or success
            reward = float(success)
            if fail and self.break_on_fail:
                import ipdb

                ipdb.set_trace()
            if done:
                info.update(
                    instruction_len=len(blobs),
                )
                if not use_failure_buf:
                    info.update(success_without_failure_buf=success)
                if success:
                    info.update(success_line=len(blobs), progress=1)

            def render():
                if done:
                    print(GREEN if success else RED)
                indent = 0
                for i, line in enumerate(blobs):
                    pre = "- " if i == ptr else "  "
                    print("{:2}{}{}{}".format(i, pre, " " * indent, str(line)))
                    indent += line.depth_change[1]
                print(RESET)
                print("Action:", action)
                print("Reward", reward)
                print("Obs:")
                grid_size = 3  # obs.astype(int).sum(-1).max()  # max objects per grid
                chars = [" "] + [x.name[0] for x in WorldObjects]
                print(self.i)
                pprint(inventory)
                for i, row in enumerate(room):
                    colors = []
                    string = []
                    for j, channel in enumerate(row):
                        int_ids = 1 + np.arange(channel.size)
                        number = channel * int_ids
                        crop = sorted(number, reverse=True)[:grid_size]
                        for x in crop:
                            import ipdb

                            ipdb.set_trace()
                            colors.append(Colors[WorldObjects[x - 1]])
                            string.append(chars[x])
                        colors.append(RESET)
                        string.append("|")
                    print(*[c for p in zip(colors, string) for c in p], sep="")
                    print("-" * len(string))

            self._render = render

            # feasibility
            counts = Counter(objects.values())
            if any(counts[o] == 0 for o in Necessary):

                def get_free_space():
                    for i in range(self.h):
                        for j in range(self.w):
                            maybe_object = objects.get((i, j), None)
                            if (
                                maybe_object not in self.limina
                                and counts[maybe_object] > 1
                            ):
                                yield i, j

                free_space = list(get_free_space())
                assert free_space

                for o in Necessary:
                    if counts[o] == 0:
                        coord = free_space[self.random.choice(len(free_space))]
                        objects[coord] = o

            # obs
            room = np.zeros(self.room_shape)
            for p, o in list(objects.items()):
                p = np.array(p)
                room[tuple((o.value, *p))] = 1
            room[tuple((len(WorldObjects), *agent_pos))] = 1

            inventory += inventory_change
            inventory = {k: min(v, self.max_inventory) for k, v in inventory.items()}

            obs = Obs(
                obs=rooms,
                lines=preprocess_lines(),
                mask=[p is None for p in padded],
                inventory=self.inventory_representation(inventory),
                inventory_change=self.inventory_representation(inventory_change),
            )

            inventory_change = Counter()

            # if not self.observation_space.contains(obs):
            #     import ipdb
            #
            #     ipdb.set_trace()
            #     self.observation_space.contains(obs)
            obs = OrderedDict(obs._asdict())

            line_specific_info = {
                f"{k}_{10 * (len(blobs) // 10)}": v for k, v in info.items()
            }
            action = (
                yield obs,
                reward,
                done,
                dict(**info, **line_specific_info),
            )
            if action.size == 1:
                action = Action(upper=0, lower=action, delta=0, dg=0, ptr=0)
            action = Action(*action)
            action = action._replace(lower=self.lower_level_actions[action.lower])

            info = dict(
                use_failure_buf=use_failure_buf,
                len_failure_buffer=len(self.failure_buffer),
            )

            if no_op(action.upper):
                no_op_count += 1
                no_op_limit = 200 if self.evaluating else self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(blobs)
                if no_op_count >= no_op_limit:
                    done = True
                    continue

            if self.random.random() < self.bandit_prob:
                robbed = self.random.choice([k for k, v in inventory.items() if v > 0])
                inventory_change[robbed] = -1

            if isinstance(action.lower, np.ndarray):
                new_pos = (agent_pos + action.lower) % self.room_shape
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
                        for required_resource in required:
                            if (
                                inventory[required_resource]
                                < required[required_resource]
                            ):
                                return False
                        return True
                    if moving_into == Terrain.MOUNTAIN:
                        return bool(inventory[Other.MAP])
                    return True

                if valid_new_pos():
                    if moving_into == Terrain.WATER:
                        inventory_change -= required  # build bridge
                        if self.random.random() > self.bridge_failure_prob:
                            agent_pos = new_pos  # check for "flash flood"
                        # else bridge failed
                    else:
                        agent_pos = new_pos
                        if moving_into == Terrain.MOUNTAIN:
                            inventory_change[Other.MAP] = -inventory[Other.MAP]
                        if next_room():
                            room = next(rooms_iter)
                            objects = dict(next(blobs_iter))
            elif type(action.lower) is str:
                standing_on = objects.get(tuple(agent_pos), None)
                if action.lower == Interaction.COLLECT and standing_on is not None:
                    inventory_change[standing_on] = 1 + self.random.choice(
                        2, p=self.windfall_prob
                    )
                    del objects[tuple(agent_pos)]
                    if self.random.random() < self.map_discovery_prob:
                        inventory_change[Other.MAP] = 1
                if (
                    action.lower[0] == Interaction.REFINE
                    and standing_on == Terrain.FACTORY
                ):
                    item = action.lower[1]
                    if inventory[item]:
                        inventory_change[item] = -1
                        inventory_change[Refined(item.value)] = 1
            else:
                raise RuntimeError

    def inventory_representation(self, inventory):
        return np.array([inventory[k] for k in InventoryItems])

    def seed(self, seed=None):
        assert self.seed == seed

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("pause")


def add_arguments(p):
    p.add_argument("--min-lines", type=int)
    p.add_argument("--max-lines", type=int)
    p.add_argument("--no-op-limit", type=int)
    p.add_argument("--subtasks-only", action="store_true")
    p.add_argument("--break-on-fail", action="store_true")
    p.add_argument("--time-to-waste", type=int)
    p.add_argument("--max-failure-sample-prob", type=float)
    p.add_argument("--failure-buffer-size", type=int)
    p.add_argument("--room-size", type=int)


def main(env):
    # for i, l in enumerate(env.lower_level_actions):
    # print(i, l)
    actions = [x if type(x) is str else tuple(x) for x in env.lower_level_actions]
    mapping = dict(
        w=(-1, 0), s=(1, 0), a=(0, -1), d=(0, 1), m="mine", l="sell", g="goto"
    )
    mapping2 = {}
    for k, v in mapping.items():
        try:
            mapping2[k] = actions.index(v)
        except ValueError:
            pass

    def action_fn(string):
        action = mapping2.get(string, None)
        if action is None:
            return None
        return np.array(Action(upper=0, lower=action, delta=0, dg=0, ptr=0))

    keyboard_control.run(env, action_fn=action_fn)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--min-eval-lines", type=int)
    PARSER.add_argument("--max-eval-lines", type=int)
    add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    main(Env(rank=0, lower_level="train-alone", **hierarchical_parse_args(PARSER)))
