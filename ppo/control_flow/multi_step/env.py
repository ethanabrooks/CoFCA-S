import functools
from collections import defaultdict, Counter
from typing import Iterator, List, Tuple, Generator, Dict, Union, Optional

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo import keyboard_control
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


class Env(ppo.control_flow.env.Env):
    wood = "wood"
    gold = "gold"
    iron = "iron"
    merchant = "merchant"
    bridge = "bridge"
    agent = "agent"
    mine = "mine"
    sell = "sell"
    goto = "goto"
    items = [wood, gold, iron, merchant]
    terrain = [bridge, agent]
    world_contents = items + terrain
    behaviors = [mine, sell, goto]

    def __init__(
        self,
        num_subtasks,
        max_while_objects,
        num_excluded_objects,
        temporal_extension,
        world_size=6,
        **kwargs,
    ):
        self.num_excluded_objects = num_excluded_objects
        self.max_while_objects = max_while_objects
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

        self.action_space = spaces.MultiDiscrete(
            np.array([num_subtasks + 1, 2 * self.n_lines, 2, 2, self.n_lines])
        )
        self.observation_space.spaces.update(
            obs=spaces.Box(low=0, high=1, shape=self.world_shape),
            lines=spaces.MultiDiscrete(
                np.array(
                    [
                        [
                            len(self.line_types),
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
            return [self.line_types.index(type(line)), 0, 0, 0]
        elif type(line) is Loop:
            return [self.line_types.index(Loop), 0, 0, line.id]
        elif type(line) is Subtask:
            i, o = line.id
            i, o = self.behaviors.index(i), self.items.index(o)
            return [self.line_types.index(Subtask), i + 1, o + 1, 0]
        elif type(line) in (While, If):
            return [
                self.line_types.index(type(line)),
                0,
                self.items.index(line.id) + 1,
                0,
            ]
        else:
            raise RuntimeError()

    def world_array(self, object_pos, agent_pos):
        world = np.zeros(self.world_shape)
        for o, p in object_pos + [(self.agent, agent_pos)]:
            p = np.array(p)
            world[tuple((self.world_contents.index(o), *p))] = 1
        return world

    @staticmethod
    def evaluate_line(line, object_pos, condition_evaluations, loops):
        if line is None:
            return None
        elif type(line) is Loop:
            return loops > 0
        if type(line) is Subtask:
            return 1
        else:
            evaluation = any(o == line.id for o, _ in object_pos)
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
            object_pos = []
            assert self.max_nesting_depth == 1
            agent_pos = self.random.randint(0, self.world_size, size=2)
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
                object_pos.append((o, p))

            line_iterator = self.line_generator(lines)
            condition_evaluations = []
            self.time_remaining = 200 if self.evaluating else self.time_to_waste
            self.loops = None

            def get_nearest(to):
                candidates = [np.array(p) for o, p in object_pos if o == to]
                if candidates:
                    return min(candidates, key=lambda k: np.sum(np.abs(agent_pos - k)))

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
                                lines[l], object_pos, condition_evaluations, self.loops
                            )
                        )
                        if self.loops == 0:
                            self.loops = None
                    if l is None or type(lines[l]) is Subtask:
                        break
                if l is not None:
                    assert type(lines[l]) is Subtask
                    _, o = lines[l].id
                    n = get_nearest(o)
                    if n is not None:
                        self.time_remaining += 1 + np.max(np.abs(agent_pos - n))
                return l

            possible_objects = [o for o, _ in object_pos]
            prev, ptr = 0, next_subtask(None)
            term = False
            while True:
                term |= not self.time_remaining
                subtask_id = yield State(
                    obs=self.world_array(object_pos, agent_pos),
                    prev=prev,
                    ptr=ptr,
                    term=term,
                )
                self.time_remaining -= 1
                interaction, obj = self.subtasks[subtask_id]

                def pair():
                    return obj, tuple(agent_pos)

                def on_object():
                    return pair() in object_pos  # standing on the desired object

                correct_id = (interaction, obj) == lines[ptr].id
                if on_object():
                    if interaction in (self.mine, self.sell):
                        object_pos.remove(pair())
                        if correct_id:
                            possible_objects.remove(obj)
                        else:
                            term = True
                    if interaction == self.sell:
                        object_pos.append((self.bridge, tuple(agent_pos)))
                    if correct_id:
                        prev, ptr = ptr, next_subtask(ptr)
                else:
                    nearest = get_nearest(obj)
                    if nearest is not None:
                        delta = nearest - agent_pos
                        if self.temporal_extension:
                            delta = np.clip(delta, -1, 1)
                        agent_pos += delta
                    elif correct_id and obj not in possible_objects:
                        # subtask is impossible
                        prev, ptr = ptr, None

        return state_generator(), lines

    def populate_world(self, lines):
        line_io = [line.id for line in lines if type(line) is Subtask]
        line_pos = self.random.randint(0, self.world_size, size=(len(line_io), 2))
        object_pos = [
            (o, tuple(pos)) for (interaction, o), pos in zip(line_io, line_pos)
        ]
        while_blocks = defaultdict(list)  # while line: child subtasks
        active_whiles = []
        active_loops = []
        loop_obj = []
        loop_count = 0
        for i, line in enumerate(lines):
            if type(line) is While:
                active_whiles += [i]
            elif type(line) is EndWhile:
                active_whiles.pop()
            elif type(line) is Loop:
                active_loops += [line]
            elif type(line) is EndLoop:
                active_loops.pop()
            elif type(line) is Subtask:
                if active_whiles:
                    while_blocks[active_whiles[-1]] += [i]
                if active_loops:
                    _i, _o = line.id
                    loop_num = active_loops[-1].id
                    loop_obj += [(_o, loop_num)]
                    loop_count += loop_num

        pos = self.random.randint(0, self.world_size, size=(loop_count, 2))
        obj = (o for o, c in loop_obj for _ in range(c))
        object_pos += [(o, tuple(p)) for o, p in zip(obj, pos)]

        for while_line, block in while_blocks.items():
            obj = lines[while_line].id
            l = self.random.choice(block)
            i = self.random.choice(2)
            assert self.behaviors[i] in (self.mine, self.sell)
            line_id = self.behaviors[i], obj
            assert line_id in ((self.mine, obj), (self.sell, obj))
            lines[l] = Subtask(line_id)
            if not self.evaluating and obj in self.world_contents:
                num_obj = self.random.randint(self.max_while_objects + 1)
                if num_obj:
                    pos = self.random.randint(0, self.world_size, size=(num_obj, 2))
                    object_pos += [(obj, tuple(p)) for p in pos]

        return object_pos

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


def build_parser(p):
    ppo.control_flow.env.build_parser(p)
    p.add_argument("--max-while-objects", type=float, default=2)
    p.add_argument("--num-excluded-objects", type=int, default=2)
    p.add_argument(
        "--no-temporal-extension", dest="temporal_extension", action="store_false"
    )
    return p


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = build_parser(parser)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    ppo.control_flow.env.main(Env(rank=0, **hierarchical_parse_args(parser)))
