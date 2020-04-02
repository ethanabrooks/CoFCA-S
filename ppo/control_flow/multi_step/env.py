import functools
import itertools
from collections import defaultdict
from typing import Iterator, List, Tuple

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo.control_flow.env import State
from ppo.control_flow.lines import (
    Subtask,
    Padding,
    Line,
    While,
    If,
    EndWhile,
    Else,
    EndIf,
    Loop,
    EndLoop,
)
from ppo.djikstra import shortest_path

BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
ORANGE = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
LIGHTGREY = "\033[37m"
DARKGREY = "\033[90m"
LIGHTRED = "\033[91m"
LIGHTGREEN = "\033[92m"
YELLOW = "\033[93m"
LIGHTBLUE = "\033[94m"
PINK = "\033[95m"
LIGHTCYAN = "\033[96m"
RESET = "\033[0m"


def get_nearest(_from, _to, objects):
    items = [(np.array(p), o) for p, o in objects.items()]
    candidates = [(p, np.sum(np.abs(_from - p))) for p, o in items if o == _to]
    if candidates:
        return min(candidates, key=lambda c: c[1])


class Env(ppo.control_flow.env.Env):
    wood = "wood"
    gold = "gold"
    iron = "iron"
    merchant = "merchant"
    bridge = "bridge"
    water = "stream"
    wall = "obstruction"
    agent = "agent"
    mine = "mine"
    sell = "sell"
    goto = "goto"
    items = [wood, gold, iron, merchant]
    terrain = [water, wall, bridge, agent]
    world_contents = items + terrain
    behaviors = [mine, sell, goto]

    def __init__(
        self,
        max_while_objects,
        num_subtasks,
        num_excluded_objects,
        temporal_extension,
        world_size=6,
        **kwargs,
    ):
        self.temporal_extension = temporal_extension
        self.num_excluded_objects = num_excluded_objects
        self.max_while_objects = max_while_objects
        self.loops = None
        self.i = 0

        def subtasks():
            for obj in self.items:
                for interaction in self.behaviors:
                    yield interaction, obj

        self.subtasks = list(subtasks())
        num_subtasks = len(self.subtasks)
        super().__init__(num_subtasks=num_subtasks, **kwargs)
        self.world_size = world_size
        self.world_shape = (len(self.world_contents), self.world_size, self.world_size)

        self.action_space = spaces.MultiDiscrete(
            np.array([num_subtasks + 1, 2 * self.n_lines, 2, 2, self.n_lines])
        )
        self.observation_space.spaces.update(
            obs=spaces.Box(low=0, high=1, shape=self.world_shape),
            lines=spaces.MultiDiscrete(
                np.array(
                    [
                        [
                            len(Line.types),
                            1 + len(self.behaviors),
                            1 + len(self.items),
                            1 + self.max_loops,
                        ]
                    ]
                    * self.n_lines
                )
            ),
        )

    def print_obs(self, obs):
        obs = obs.transpose(1, 2, 0).astype(int)
        grid_size = 2  # obs.astype(int).sum(-1).max()  # max objects per grid
        chars = [" "] + [o for o, *_ in self.world_contents]
        for i, row in enumerate(obs):
            string = ""
            for j, channel in enumerate(row):
                int_ids = 1 + np.arange(channel.size)
                number = channel * int_ids
                crop = sorted(number, reverse=True)[:grid_size]
                string += "".join(chars[x] for x in crop) + "|"
            print(string)
            print("-" * len(string))

    def line_str(self, line):
        line = super().line_str(line)
        if type(line) is Subtask:
            return f"{line} {self.subtasks.index(line.id)}"
        return line

    @functools.lru_cache(maxsize=200)
    def preprocess_line(self, line):
        if type(line) in (Else, EndIf, EndWhile, EndLoop, Padding):
            return [Line.types.index(type(line)), 0, 0, 0]
        elif type(line) is Loop:
            return [Line.types.index(Loop), 0, 0, line.id]
        elif type(line) is Subtask:
            i, o = line.id
            i, o = self.behaviors.index(i), self.items.index(o)
            return [Line.types.index(Subtask), i + 1, o + 1, 0]
        elif type(line) in (While, If):
            return [Line.types.index(type(line)), 0, self.items.index(line.id) + 1, 0]
        else:
            raise RuntimeError()

    def world_array(self, objects, agent_pos):
        world = np.zeros(self.world_shape)
        for p, o in list(objects.items()) + [(agent_pos, self.agent)]:
            p = np.array(p)
            world[tuple((self.world_contents.index(o), *p))] = 1

        return world

    @staticmethod
    def evaluate_line(line, objects, condition_evaluations, loops):
        if line is None:
            return None
        elif type(line) is Loop:
            return loops > 0
        if type(line) is Subtask:
            return 1
        else:
            evaluation = line.id in objects.values()
            if type(line) in (If, While):
                condition_evaluations += [evaluation]
            return evaluation

    def generators(self) -> Tuple[Iterator[State], List[Line]]:
        line_types = self.choose_line_types()
        lines = list(self.assign_line_ids(line_types))

        def state_generator() -> State:
            assert self.max_nesting_depth == 1
            objects = self.populate_world(lines)
            agent_pos = next(p for p, o in objects.items() if o == self.agent)

            line_iterator = self.line_generator(lines)
            condition_evaluations = []
            self.time_remaining = 200 if self.evaluating else self.time_to_waste
            self.loops = None

            def next_subtask(l):
                while True:
                    if l is None:
                        l = line_iterator.send(None)
                    else:
                        if type(lines[l]) is Loop:
                            if self.loops is None:
                                self.loops = lines[l].id
                            else:
                                self.loops -= 1
                        l = line_iterator.send(
                            self.evaluate_line(
                                lines[l], objects, condition_evaluations, self.loops
                            )
                        )
                        if self.loops == 0:
                            self.loops = None
                    if l is None or type(lines[l]) is Subtask:
                        break
                if l is not None:
                    assert type(lines[l]) is Subtask
                    _, o = lines[l].id
                    nearest = get_nearest(_to=o, _from=agent_pos, objects=objects)
                    if nearest is None:
                        return None
                    _, d = nearest
                    self.time_remaining += 1 + d
                    return l

            possible_objects = list(objects.values())
            prev, ptr = 0, next_subtask(None)
            term = False
            while True:
                term |= not self.time_remaining
                subtask_id = yield State(
                    obs=self.world_array(objects, agent_pos),
                    prev=prev,
                    ptr=ptr,
                    term=term,
                )
                self.time_remaining -= 1
                interaction, obj = self.subtasks[subtask_id]

                def pair():
                    return obj, tuple(agent_pos)

                def on_object():
                    try:
                        return objects[tuple(agent_pos)] == obj
                    except KeyError:
                        return False

                correct_id = (interaction, obj) == lines[ptr].id

                lower_level_action = self.get_lower_level_action(
                    interaction, obj, tuple(agent_pos), objects
                )
                if on_object():
                    if (
                        type(lower_level_action) is str
                        and lower_level_action == self.mine
                    ):
                        del objects[tuple(agent_pos)]
                        if correct_id:
                            possible_objects.remove(obj)
                        else:
                            term = True
                    if correct_id:
                        prev, ptr = ptr, next_subtask(ptr)
                else:
                    if type(lower_level_action) is np.ndarray:
                        if self.temporal_extension:
                            lower_level_action = np.clip(lower_level_action, -1, 1)
                        agent_pos += lower_level_action
                    elif correct_id and obj not in possible_objects:
                        assert obj not in possible_objects
                        # subtask is impossible
                        prev, ptr = ptr, None

        return state_generator(), lines

    def populate_world(self, lines):
        def subtask_ids():
            for line in lines:
                if type(line) is Subtask:
                    _i, _o = line.id
                    yield _o

        def loop_objects():
            active_loops = []
            for i, line in enumerate(lines):
                if type(line) is Loop:
                    active_loops += [line]
                elif type(line) is EndLoop:
                    active_loops.pop()
                elif type(line) is Subtask:
                    if active_loops:
                        _i, _o = line.id
                        loop_num = active_loops[-1].id
                        for _ in range(loop_num):
                            yield _o

        def while_objects():
            while_blocks = defaultdict(list)  # while line: child subtasks
            active_whiles = []
            for i, line in enumerate(lines):
                if type(line) is While:
                    active_whiles += [i]
                elif type(line) is EndWhile:
                    active_whiles.pop()
                elif type(line) is Subtask:
                    if active_whiles:
                        while_blocks[active_whiles[-1]] += [i]

            for while_line, block in while_blocks.items():
                obj = lines[while_line].id
                l = self.random.choice(block)
                line_id = self.mine, obj
                lines[l] = Subtask(line_id)
                if not self.evaluating and obj in self.world_contents:
                    num_obj = self.random.randint(self.max_while_objects + 1)
                    for _ in range(num_obj):
                        yield obj

        vertical_water = self.random.choice(2)
        world_shape = (
            [self.world_size, self.world_size - 1]
            if vertical_water
            else [self.world_size - 1, self.world_size]
        )

        object_list = (
            [self.agent]
            + list(subtask_ids())
            + list(loop_objects())
            + list(while_objects())
        )
        num_random_objects = (self.world_size ** 2) - self.world_size  # for water
        object_list = object_list[:num_random_objects]
        indexes = self.random.choice(
            self.world_size * (self.world_size - 1),
            size=num_random_objects,
            replace=False,
        )
        positions = np.array(list(zip(*np.unravel_index(indexes, world_shape))))
        wall_indexes = positions[:, 0] % 2 * positions[:, 1] % 2
        wall_positions = positions[wall_indexes == 1]
        object_positions = positions[wall_indexes == 0]
        num_walls = self.random.choice(len(wall_positions))
        object_positions = object_positions[: len(object_list)]
        if len(object_list) == len(object_positions):
            wall_positions = wall_positions[:num_walls]
        positions = np.concatenate([object_positions, wall_positions])
        water_index = self.random.choice(self.world_size)
        positions[positions[:, vertical_water] >= water_index] += np.array(
            [0, 1] if vertical_water else [1, 0]
        )
        assert water_index not in positions[:, vertical_water]
        return {
            **{
                tuple(p): (self.wall if o is None else o)
                for o, p in itertools.zip_longest(object_list, positions)
            },
            **{
                (i, water_index) if vertical_water else (water_index, i): self.water
                for i in range(self.world_size)
            },
        }

    def assign_line_ids(self, lines):
        excluded = self.random.randint(len(self.items), size=self.num_excluded_objects)
        included_objects = [o for i, o in enumerate(self.items) if i not in excluded]

        interaction_ids = self.random.choice(len(self.behaviors), size=len(lines))
        object_ids = self.random.choice(len(included_objects), size=len(lines))
        line_ids = self.random.choice(len(self.items), size=len(lines))

        for line, line_id, interaction_id, object_id in zip(
            lines, line_ids, interaction_ids, object_ids
        ):
            if line is Subtask:
                subtask_id = (
                    self.behaviors[interaction_id],
                    included_objects[object_id],
                )
                yield Subtask(subtask_id)
            elif line is Loop:
                yield Loop(self.random.randint(1, 1 + self.max_loops))
            else:
                yield line(self.items[line_id])

    def get_lower_level_action(self, interaction, obj, agent_pos, objects):
        if interaction == self.sell:
            obj = self.merchant
        if objects.get(tuple(agent_pos), None) == obj:
            return interaction
        else:
            nearest = get_nearest(_from=agent_pos, _to=obj, objects=objects)
            if nearest:
                n, d = nearest
                return n - agent_pos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = build_parser(parser)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    ppo.control_flow.env.main(Env(rank=0, **hierarchical_parse_args(parser)))
