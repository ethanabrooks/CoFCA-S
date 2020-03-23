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
    line_types = list(Line.types)

    def __init__(self, num_subtasks, temporal_extension, world_size=6, **kwargs):
        self.temporal_extension = temporal_extension
        self.loops = None

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
                            len(self.control_flow_types),
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

    def generators(self,) -> Tuple[Iterator[State], List[Line]]:
        line_types = self.choose_line_types()

        # if there are too many while loops,
        # all items will be eliminated and tasks become impossible
        while_count = 0
        for line_type in line_types:
            if line_type is EndWhile:
                while_count += 1
            if while_count >= len(self.items) - 1:
                # too many while_loops
                return self.generators()
        assert not while_count >= len(self.items) - 1  # TODO

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

        blocks = defaultdict(list)
        whiles = []
        for i, line_type in enumerate(line_types):
            if line_type is While:
                whiles.append(i)
            elif line_type is EndWhile:
                whiles.pop()
            else:
                for w in whiles:
                    blocks[w].append(i)

        lines = [l(None) for l in line_types]  # instantiate line types

        # select line inside while blocks to be a build behavior
        # so as to prevent infinite while loops
        while_index = {}  # type: Dict[int, line]
        for i, indices in blocks.items():
            indices = set(indices) - set(while_index.keys())
            while_index[self.random.choice(list(indices))] = lines[i]

        # go through lines in reverse to assign ids and put objects in the world
        existing = list(
            self.random.choice(
                self.items, size=len(self.items) - line_types.count(While) - 1
            )
        )
        non_existing = list(set(self.items) - set(existing))
        world = Counter()
        index_truthiness = list(index_truthiness_generator())
        for i, truthy in reversed(index_truthiness):
            line = lines[i]
            if type(line) is Subtask:
                if not line.id:
                    subtasks = [s for s in self.subtasks]
                    try:
                        item = while_index[i].id
                        assert item is not None
                        behavior = self.mine
                        subtasks.remove((self.mine, item))
                    except KeyError:
                        behavior, item = subtasks[self.random.choice(len(subtasks))]
                    line.id = (behavior, item)
                behavior, item = line.id
                if not world[item] or behavior in [self.mine, self.sell]:
                    world[item] += 1
            elif type(line) is If:
                if not line.id:
                    line.id = self.random.choice(existing if truthy else non_existing)
                if truthy and not world[line.id]:
                    world[line.id] += 1
            elif type(line) is While:
                if not line.id:
                    line.id = self.random.choice(non_existing)  # type: str
                if truthy and not line.id in existing:
                    existing.append(line.id)
                    non_existing.remove(line.id)
            elif type(line) is Loop:
                if line.id is None:
                    line.id = 0
                else:
                    line.id += 1
            else:
                line.id = 0
            if sum(world.values()) > self.world_size ** 2:
                # can't fit all objects on map
                return self.generators()

        # assign unvisited lines
        for line in lines:
            if line.id is None:
                line.id = self.subtasks[self.random.choice(len(self.subtasks))]

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

            time_remaining = 200 if self.evaluating else self.time_to_waste

            def subtask_generator() -> Generator[int, None, None]:
                for l, _ in index_truthiness:
                    if type(lines[l]) is Subtask:
                        yield l

            def get_nearest(obj: str) -> np.ndarray:
                candidates = [np.array(p) for p, o in objects.items() if obj == o]
                if candidates:
                    return min(candidates, key=lambda k: np.sum(np.abs(pos - k)))

            def agent_generator() -> Generator[
                Union[np.ndarray, str], Tuple[str, str], None
            ]:
                subtask, objective, move = None, None, None
                while True:
                    new_subtask = (chosen_behavior, chosen_object) = yield move
                    if new_subtask != subtask:
                        subtask = new_subtask
                        objective = get_nearest(chosen_object)
                    if objective is None:
                        move = None
                    elif np.all(objective == pos):
                        move = chosen_behavior
                        subtask = None
                    else:
                        move = np.array(objective) - pos
                        if self.temporal_extension:
                            move = np.clip(move, -1, 1)

            def world_array() -> np.ndarray:
                array = np.zeros(self.world_shape)
                for p, o in objects.items():
                    if o is not None:
                        p = np.array(p)
                        array[tuple((self.world_contents.index(o), *p))] = 1
                array[tuple((self.world_contents.index(self.agent), *pos))] = 1
                return array

            term = False

            def next_subtask() -> Optional[int]:
                return next(subtask_iterator, None)

            subtask_iterator = subtask_generator()
            agent_iterator = agent_generator()

            prev, ptr = 0, next_subtask()
            if ptr is not None:
                _, o = lines[ptr].id
                nearest = get_nearest(o)
                if nearest is None:
                    import ipdb

                    ipdb.set_trace()
                time_remaining += max(np.abs(nearest - pos)) + 1
            next(agent_iterator)
            while True:
                term |= not time_remaining
                subtask_id = yield State(
                    obs=world_array(), prev=prev, ptr=ptr, term=term
                )
                time_remaining -= 1
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
                    if ptr is not None:
                        _, o = lines[ptr].id
                        nearest = get_nearest(o)
                        if nearest is None:
                            term = True
                        else:
                            time_remaining += max(np.abs(nearest - pos)) + 1

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
                    objects[tuple(pos)] = None
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
