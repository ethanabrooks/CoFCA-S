import copy
import functools
import itertools
from collections import defaultdict, Counter

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo import keyboard_control
from ppo.control_flow.env import State
from ppo.control_flow.lines import Subtask, Padding, Line, While, If, EndWhile, Else


class Env(ppo.control_flow.env.Env):
    wood = "wood"
    gold = "gold"
    iron = "iron"
    merchant = "merchant"
    bridge = "bridge"
    agent = "agent"
    mine = "mine"
    build = "build"
    visit = "visit"
    objects = [wood, gold, iron, merchant]
    other_objects = [bridge, agent]
    world_objects = objects + other_objects
    interactions = [mine, build, visit]

    def __init__(
        self,
        max_while_objects,
        num_subtasks,
        num_excluded_objects,
        temporal_extension,
        world_size,
        **kwargs,
    ):
        self.num_objects = 20
        self.temporal_extension = temporal_extension
        self.num_excluded_objects = num_excluded_objects
        self.max_while_objects = max_while_objects

        def subtasks():
            for obj in self.objects:
                for interaction in self.interactions:
                    yield interaction, obj

        self.subtasks = list(subtasks())
        num_subtasks = len(self.subtasks)
        super().__init__(num_subtasks=num_subtasks, **kwargs)
        self.world_size = world_size
        self.world_shape = (len(self.world_objects), self.world_size, self.world_size)

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
                            1 + len(self.interactions) + len(self.objects),
                            1 + len(self.objects),
                        ]
                    ]
                    * self.n_lines
                )
            ),
        )

    def line_str(self, line: Line):
        if isinstance(line, Subtask):
            return f"Subtask {self.subtasks.index(line.id)}: {line.id}"
        elif isinstance(line, (If, While)):
            o1, o2 = line.id
            return f"{line}: count[{o1}] < count[{o2}]"
        else:
            return f"{line}"

    def print_obs(self, obs):
        obs = obs.transpose(1, 2, 0).astype(int)
        grid_size = obs.astype(int).sum(-1).max()  # max objects per grid
        chars = [" "] + [o for o, *_ in self.world_objects]
        for i, row in enumerate(obs):
            string = ""
            for j, channel in enumerate(row):
                int_ids = 1 + np.arange(channel.size)
                number = channel * int_ids
                crop = sorted(number, reverse=True)[:grid_size]
                string += "".join(chars[x] for x in crop) + "|"
            print(string)
            print("-" * len(string))

    @functools.lru_cache(maxsize=200)
    def preprocess_line(self, line):
        if type(line) is Subtask:
            i, o = line.id
            i, o = self.interactions.index(i), self.objects.index(o)
            return [self.line_types.index(Subtask), i + 1, o + 1]
        elif type(line) in (While, If):
            o1, o2 = line.id
            return [
                self.line_types.index(type(line)),
                1 + len(self.interactions) + self.objects.index(o1),
                self.objects.index(o2) + 1,
            ]
        else:
            return [self.line_types.index(type(line)), 0, 0]

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
        elif type(line) in (While, If):
            o1, o2 = line.id
            pos_obj = defaultdict(set)
            for o, p in object_pos:
                pos_obj[p].add(o)

            count1 = sum(1 for _, ob_set in pos_obj.items() if o1 in ob_set)
            count2 = sum(1 for _, ob_set in pos_obj.items() if o2 in ob_set)
            evaluation = count1 < count2
            if type(line) in (If, While):
                condition_evaluations[type(line)] += [evaluation]
            return evaluation
        else:
            return 1

    def state_generator(self, lines) -> State:
        assert self.max_nesting_depth == 1
        agent_pos = self.random.randint(0, self.world_size, size=2)
        object_pos, lines = self.populate_world(lines)
        line_iterator = self.line_generator(lines)
        condition_evaluations = defaultdict(list)
        self.time_remaining = self.time_to_waste

        def get_nearest(to):
            candidates = [np.array(p) for o, p in object_pos if o == to]
            if candidates:
                return min(candidates, key=lambda k: np.sum(np.abs(agent_pos - k)))

        def next_subtask(l):
            if l is None:
                l = line_iterator.send(None)
            else:
                l = line_iterator.send(
                    self.evaluate_line(lines[l], object_pos, condition_evaluations)
                )
            while not (l is None or type(lines[l]) is Subtask):
                l = line_iterator.send(
                    self.evaluate_line(lines[l], object_pos, condition_evaluations)
                )
            if l is not None:
                assert type(lines[l]) is Subtask
                _, o = lines[l].id
                n = get_nearest(o)
                if n is not None:
                    self.time_remaining += 1
                    if self.temporal_extension:
                        self.time_remaining += np.max(np.abs(agent_pos - n))
            return l

        possible_objects = [o for o, _ in object_pos]
        prev, ptr = 0, next_subtask(None)
        term = False
        while True:
            term |= not self.time_remaining
            subtask_id = yield State(
                obs=self.world_array(object_pos, agent_pos),
                condition=None,
                prev=prev,
                ptr=ptr,
                condition_evaluations=condition_evaluations,
                term=term,
            )
            self.time_remaining -= 1
            interaction, obj = self.subtasks[subtask_id]

            def pair():
                return obj, tuple(agent_pos)

            def on_object():
                return pair() in object_pos  # standing on the desired object

            correct_id = (interaction, obj) == lines[ptr].id
            if not self.temporal_extension:
                if correct_id and obj not in possible_objects:
                    # subtask is impossible
                    prev, ptr = ptr, None
                else:
                    nearest = get_nearest(obj)
                    if nearest is not None:
                        nearest = tuple(nearest)
                        if interaction in (self.mine, self.build):
                            object_pos.remove((obj, nearest))
                            if correct_id:
                                possible_objects.remove(obj)
                            else:
                                term = True
                        if interaction == self.build:
                            object_pos.append((self.bridge, nearest))
                        if correct_id:
                            prev, ptr = ptr, next_subtask(ptr)

            elif on_object():
                if interaction in (self.mine, self.build):
                    object_pos.remove(pair())
                    if correct_id:
                        possible_objects.remove(obj)
                    else:
                        term = True
                if interaction == self.build:
                    object_pos.append((self.bridge, tuple(agent_pos)))
                if correct_id:
                    prev, ptr = ptr, next_subtask(ptr)
            else:
                nearest = get_nearest(obj)
                if nearest is not None:
                    delta = nearest - agent_pos
                    delta = np.clip(delta, -1, 1)
                    agent_pos += delta
                elif correct_id and obj not in possible_objects:
                    # subtask is impossible
                    prev, ptr = ptr, None

    def populate_world(self, lines):
        while_blocks = defaultdict(list)  # while line: child subtasks
        active_whiles = []
        for interaction, line in enumerate(lines):
            if type(line) is While:
                active_whiles += [interaction]
            elif type(line) is EndWhile:
                active_whiles.pop()
            elif active_whiles and type(line) is Subtask:
                while_blocks[active_whiles[-1]] += [interaction]
        for while_line, block in while_blocks.items():
            o1, o2 = lines[while_line].id
            l = self.random.choice(block)
            i = self.random.choice(2)
            line_id = (self.mine, self.build)[i], o2
            lines[l] = Subtask(line_id)

        more_lines = self.get_lines(
            n=20 - len(lines),
            active_conditions=[],
            max_nesting_depth=self.max_nesting_depth,
        )
        more_lines = list(self.assign_line_ids(more_lines))
        objects = [line.id[1] for line in lines + more_lines if type(line) is Subtask]
        for while_line, block in while_blocks.items():
            o1, o2 = lines[while_line].id
            if self.max_while_objects:
                objects += [o2] * int(self.random.choice(int(self.max_while_objects)))

        pos_arrays = self.random.randint(self.world_size, size=(len(objects), 2))
        object_pos = [(o, tuple(a)) for o, a in zip(objects, pos_arrays)]
        return object_pos, lines

    def assign_line_ids(self, line_types):
        interaction_ids = self.random.choice(
            len(self.interactions), size=len(line_types)
        )
        object_ids = self.random.choice(len(self.objects), size=len(line_types))
        alt_object_ids = self.random.choice(len(self.objects) - 1, size=len(line_types))

        for line_type, object_id, alt_object_id, interaction_id in zip(
            line_types, object_ids, alt_object_ids, interaction_ids
        ):
            if line_type is Subtask:
                subtask_id = (
                    self.interactions[interaction_id],
                    self.objects[object_id],
                )
                yield Subtask(subtask_id)
            else:
                alt_objects = [o for i, o in enumerate(self.objects) if i != object_id]
                yield line_type((alt_objects[alt_object_id], self.objects[object_id]))


def build_parser(parser):
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument(
        "--no-temporal-extension", dest="temporal_extension", action="store_false"
    )
    parser.add_argument("--max-while-objects", type=float, default=2)
    parser.add_argument("--num-excluded-objects", type=int, default=2)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER = ppo.control_flow.env.build_parser(PARSER)
    build_parser(PARSER)
    ARGS = hierarchical_parse_args(PARSER)

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(
        Env(**ARGS), action_fn=action_fn,
    )
