import copy
import functools
from collections import defaultdict, Counter
from typing import Iterator, List, Tuple, Generator, Dict

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
            return [self.line_types.index(type(line)), 0, 0, 0, 0]
        elif type(line) is Loop:
            return [self.line_types.index(Loop), 0, 0, 0, line.id]
        elif type(line) is Subtask:
            i, o = line.id
            i, o = self.behaviors.index(i), self.items.index(o)
            return [self.line_types.index(Subtask), i + 1, o + 1, 0]
        elif type(line) in (While, If):
            o1, o2 = line.id
            return [
                self.line_types.index(type(line)),
                0,
                self.items.index(line.id) + 1,
                0,
            ]
        else:
            raise RuntimeError()

    @staticmethod
    def evaluate_line(line, objects, condition_evaluations, loops):
        if line is None:
            return None
        elif type(line) is Loop:
            return loops > 0
        elif type(line) in (While, If):
            o1, o2 = line.id
            pos_obj = defaultdict(set)
            for o, p in object_pos:
                pos_obj[p].add(o)

                count1 = sum(1 for _, ob_set in pos_obj.items() if o1 in ob_set)
                count2 = sum(1 for _, ob_set in pos_obj.items() if o2 in ob_set)
                evaluation = count1 < count2

        else:
            return 1

    def world_array(self, object_pos, agent_pos):
        world = np.zeros(self.world_shape)
        for o, p in object_pos + [(self.agent, agent_pos)]:
            p = np.array(p)
            world[tuple((self.world_objects.index(o), *p))] = 1
        return world

    @staticmethod
    def evaluate_line(line, object_pos, condition_evaluations):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = build_parser(parser)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    ppo.control_flow.env.main(Env(rank=0, **hierarchical_parse_args(parser)))
