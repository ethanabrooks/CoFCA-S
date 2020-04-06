import copy
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


def objective(interaction, obj):
    if interaction == Env.sell:
        return Env.merchant
    return obj


class Env(ppo.control_flow.env.Env):
    wood = "wood"
    gold = "gold"
    iron = "iron"
    merchant = "merchant"
    bridge = "=bridge"
    water = "stream"
    wall = "#wall"
    agent = "Agent"
    mine = "mine"
    sell = "sell"
    goto = "goto"
    items = [wood, gold, iron]
    terrain = [merchant, water, wall, bridge, agent]
    world_contents = items + terrain
    behaviors = [mine, sell, goto]
    colors = [RESET, GREEN, YELLOW, LIGHTGREY, PINK, BLUE, DARKGREY, RESET, RESET]

    def __init__(self, num_subtasks, temporal_extension, world_size=6, **kwargs):
        self.temporal_extension = temporal_extension
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

        def lower_level_actions():
            yield from self.behaviors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    yield np.array([i, j])

        self.lower_level_actions = list(lower_level_actions())
        self.action_space = spaces.MultiDiscrete(
            np.array(
                [
                    num_subtasks + 1,
                    2 * self.n_lines,
                    2,
                    2,
                    len(self.lower_level_actions),
                    self.n_lines,
                ]
            )
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
        grid_size = 3  # obs.astype(int).sum(-1).max()  # max objects per grid
        chars = [" "] + [o for (o, *_) in self.world_contents]
        for i, row in enumerate(obs):
            colors = []
            string = []
            for j, channel in enumerate(row):
                int_ids = 1 + np.arange(channel.size)
                number = channel * int_ids
                crop = sorted(number, reverse=True)[:grid_size]
                for x in crop:
                    colors.append(self.colors[x])
                    string.append(chars[x])
                colors.append(RESET)
                string.append("|")
                # string += "".join(self.colors[x] + chars[x] + RESET for x in crop) + "|"
            print(*[c for p in zip(colors, string) for c in p], sep="")
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
        if type(line) is Subtask:
            return 1
        else:
            evaluation = line.id in objects.values()
            if type(line) in (If, While):
                condition_evaluations += [evaluation]
            return evaluation

    def generators(self) -> Tuple[Iterator[State], List[Line]]:
        line_types = self.choose_line_types()
        while_count = line_types.count(While)
        if while_count >= len(self.items) - 1:
            return self.generators()

        # get forward and backward transition edges
        line_transitions = defaultdict(list)
        reverse_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(line_types):
            line_transitions[_from].append(_to)
            reverse_transitions[_to].append(_from)

        # get flattened generator of index, truthy values
        def index_truthiness_generator() -> Generator[int, None, None]:
            loop_count = 0
            j = 0
            if_evaluations = []
            while j in line_transitions:
                t = self.random.choice(2)

                # make sure not to exceed max_loops
                if line_types[j] is Loop:
                    if loop_count == self.max_loops:
                        t = 0
                    if t:
                        loop_count += 1
                    else:
                        loop_count = 0
                elif line_types[j] is If:
                    if_evaluations.append(t)
                elif line_types[j] is Else:
                    t = not if_evaluations[-1]
                elif line_types[j] is EndIf:
                    if_evaluations.pop()
                yield j, t
                j = line_transitions[j][int(t)]

        index_truthiness = list(index_truthiness_generator())
        instruction_lines = [l(None) for l in line_types]  # instantiate line types

        # identify while blocks
        blocks = defaultdict(list)
        whiles = []
        for i, line_type in enumerate(line_types):
            if line_type is While:
                whiles.append(i)
            elif line_type is EndWhile:
                whiles.pop()
            else:
                for _while in whiles:
                    blocks[_while].append(i)

        # select line inside while blocks to be a build behavior
        # so as to prevent infinite while loops
        while_index = {}  # type: Dict[int, line]
        for i, indices in blocks.items():
            indices = set(indices) - set(while_index.keys())
            while_index[self.random.choice(list(indices))] = i

        # determine whether whiles are truthy (since last time we see
        # noinspection PyTypeChecker
        first_truthy = dict(reversed(index_truthiness))

        # choose existing and non_existing
        size = while_count + 1
        non_existing = list(self.random.choice(self.items, replace=False, size=size))
        existing = [o for o in self.items if o not in non_existing]

        # assign while
        for i, line in reversed(list(enumerate(instruction_lines))):
            # need to go in reverse because while can modify existing
            if type(line) is While and line.id is None:
                line.id = item = self.random.choice(non_existing)
                if item in non_existing and first_truthy[i]:
                    non_existing.remove(item)
                    # print(i, "existing", existing)

        # assign Subtask
        visited_lines = [i for i, _ in index_truthiness]
        visited_lines_set = set(visited_lines)
        for i, line in enumerate(instruction_lines):
            if type(line) is Subtask:

                if i in while_index and i in visited_lines_set:
                    behavior = self.mine
                    item = instruction_lines[while_index[i]].id
                    assert item is not None  # need to go forward to avoid this
                else:
                    behavior = self.random.choice(self.behaviors)
                    item = self.random.choice(
                        existing if i in visited_lines_set else non_existing
                    )
                line.id = (behavior, item)

        # populate world
        world = Counter()
        for i in visited_lines:
            line = instruction_lines[i]
            if type(line) is Subtask:
                behavior, item = line.id
                world[item] += 1
                if sum(world.values()) > self.world_size ** 2:
                    return self.generators()
        virtual_world = Counter(world)

        # assign other visited lines
        for i, truthy in index_truthiness:
            line = instruction_lines[i]
            if type(line) is Subtask:
                behavior, item = line.id
                if behavior in (self.mine, self.sell):
                    virtual_world[item] -= 1
            elif type(line) is If:
                # NOTE: this will not work if If is inside a loop
                if truthy:
                    sample_from = [o for o, c in virtual_world.items() if c > 0]
                    line.id = self.random.choice(sample_from)
                else:
                    sample_from = [o for o in self.items if virtual_world[o] == 0]
                    line.id = self.random.choice(sample_from)
            elif type(line) in (Else, EndIf, EndWhile, EndLoop, Padding):
                line.id = 0
            elif type(line) is Loop:
                line.id = 0 if line.id is None else line.id + 1
            else:
                assert type(line) is While
                assert line.id is not None

        # assign unvisited lines
        for line in instruction_lines:
            if line.id is None:
                line.id = self.subtasks[self.random.choice(len(self.subtasks))]
        lines = instruction_lines

        def state_generator() -> State:
            assert self.max_nesting_depth == 1
            objects = self.populate_world(lines)
            agent_pos = next(p for p, o in objects.items() if o == self.agent)
            del objects[agent_pos]

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
                    o = objective(*lines[l].id)
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
                subtask_id, lower_level_index = yield State(
                    obs=self.world_array(objects, agent_pos),
                    prev=prev,
                    ptr=ptr,
                    term=term,
                )
                # for i, a in enumerate(self.lower_level_actions):
                # print(i, a)
                # lower_level_index = int(input("go:"))
                lower_level_action = self.lower_level_actions[lower_level_index]
                self.time_remaining -= 1
                interaction, obj = self.subtasks[subtask_id]
                tgt_interaction, tgt_obj = lines[ptr].id
                tgt_obj = objective(*lines[ptr].id)

                if type(lower_level_action) is str:
                    done = (
                        lower_level_action == tgt_interaction
                        and objects.get(tuple(agent_pos), None) == tgt_obj
                    )
                    if lower_level_action == self.mine:
                        if tuple(agent_pos) in objects:
                            if done:
                                possible_objects.remove(objects[tuple(agent_pos)])
                            else:
                                term = True
                            del objects[tuple(agent_pos)]
                    if done:
                        prev, ptr = ptr, next_subtask(ptr)

                elif type(lower_level_action) is np.ndarray:
                    if self.temporal_extension:
                        lower_level_action = np.clip(lower_level_action, -1, 1)
                    new_pos = agent_pos + lower_level_action
                    if np.all(0 <= new_pos) and np.all(new_pos < self.world_size):
                        agent_pos = new_pos
                else:
                    assert lower_level_action is None

        return state_generator(), lines

    def populate_world(self, lines):
        def subtask_ids():
            for line in lines:
                if type(line) is Subtask:
                    _i, _o = line.id
                    if _i == self.sell:
                        yield self.merchant
                    else:
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
        num_walls = (
            self.random.choice(len(wall_positions)) if len(wall_positions) else 0
        )
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
                assert type(line) is While
                assert line.id is not None

        # assign unvisited lines
        for line in instruction_lines:
            if line.id is None:
                line.id = self.subtasks[self.random.choice(len(self.subtasks))]
        lines = instruction_lines

        def state_generator() -> Generator[State, int, None]:
            pos = self.random.randint(self.world_size, size=2)
            objects = {}
            flattened = [o for o, c in world.items() for _ in range(c)]
            for o, p in zip(
                flattened,
                self.random.choice(
                    self.world_size ** 2, replace=False, size=len(flattened)
                ),
            ):
                p = np.unravel_index(p, (self.world_size, self.world_size))
                objects[tuple(p)] = o

            self.time_remaining = 200 if self.evaluating else self.time_to_waste
            self.loops = None

            def subtask_generator() -> Generator[int, None, None]:
                for l, _ in index_truthiness:
                    if type(lines[l]) is Subtask:
                        yield l

            def get_nearest(to):
                candidates = [(np.array(p)) for p, o in objects.items() if o == to]
                if candidates:
                    return min(candidates, key=lambda k: np.sum(np.abs(pos - k)))

            def agent_generator() -> Generator[
                Union[np.ndarray, str], Tuple[str, str], None
            ]:
                subtask, objective, move = None, None, None
                while True:
                    (chosen_behavior, chosen_object) = yield move
                    # if new_subtask != subtask:
                    #     subtask = new_subtask
                    objective = get_nearest(chosen_object)
                    if objective is None:
                        move = None
                    elif np.all(objective == pos):
                        move = chosen_behavior
                    else:
                        move = np.array(objective) - pos
                        if self.temporal_extension:
                            move = np.clip(move, -1, 1)

            def world_array() -> np.ndarray:
                array = np.zeros(self.world_shape)
                for p, o in list(objects.items()) + [(pos, self.agent)]:
                    p = np.array(p)
                    array[tuple((self.world_contents.index(o), *p))] = 1

                return array

            subtask_iterator = subtask_generator()
            agent_iterator = agent_generator()
            next(agent_iterator)

            def next_subtask():
                try:
                    l = next(subtask_iterator)
                except StopIteration:
                    return
                if l is not None:
                    assert type(lines[l]) is Subtask
                    _, o = lines[l].id
                    n = get_nearest(o)
                    if n is not None:
                        self.time_remaining += 1 + np.max(np.abs(pos - n))
                return l

            prev, ptr = 0, next_subtask()
            term = False
            while True:
                term |= not self.time_remaining
                subtask_id = yield State(
                    obs=world_array(), prev=prev, ptr=ptr, term=term,
                )
                self.time_remaining -= 1
                chosen_subtask = self.subtasks[subtask_id]
                agent_move = agent_iterator.send(chosen_subtask)
                tgt_move, tgt_object = lines[ptr].id
                if (
                    tuple(pos) in objects
                    and tgt_object == objects[tuple(pos)]
                    and tgt_move is self.goto
                    or (agent_move is not None and tuple(agent_move) == tuple(tgt_move))
                ):
                    prev, ptr = ptr, next_subtask()

                def check_fail():
                    if objects[tuple(pos)] != tgt_object:
                        return True

                if type(agent_move) in (np.ndarray, tuple):
                    pos += agent_move
                    if np.any(pos >= self.world_size) or np.any(pos < 0):
                        import ipdb

                        ipdb.set_trace()

                elif agent_move == self.mine:
                    term = check_fail() or term
                    del objects[tuple(pos)]
                elif agent_move == self.sell:
                    term = check_fail() or term
                    objects[tuple(pos)] = self.bridge
                elif agent_move in [self.goto, None]:
                    pass
                else:
                    raise RuntimeError

        return state_generator(), lines


def build_parser(p):
    ppo.control_flow.env.build_parser(p)
    p.add_argument(
        "--no-temporal-extension", dest="temporal_extension", action="store_false"
    )
    return p

    def get_lower_level_action(self, interaction, obj, agent_pos, objects):
        obj = objective(interaction, obj)
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
