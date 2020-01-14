import functools
from collections import Counter, defaultdict
from copy import copy

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo import keyboard_control
from ppo.control_flow.env import build_parser, State
from ppo.control_flow.lines import Subtask, Padding, Line, While, If, EndWhile, Else


class Env(ppo.control_flow.env.Env):
    objects = ["pig", "sheep", "cat", "greenbot"]
    other_objects = ["ice", "agent"]
    world_objects = objects + other_objects
    mine = 0
    bridge = 1
    sell = 2
    interactions = [mine, bridge, sell]

    def __init__(
        self,
        world_size,
        max_while_objects,
        time_to_waste,
        num_subtasks,
        num_excluded_objects,
        **kwargs,
    ):
        self.num_excluded_objects = num_excluded_objects
        self.max_while_objects = max_while_objects
        self.time_to_waste = time_to_waste

        def subtasks():
            for interaction in self.interactions:
                for obj in self.objects:
                    yield interaction, obj

        self.subtask_id_to_strings = list(subtasks())
        num_subtasks = len(self.subtask_id_to_strings)
        super().__init__(num_subtasks=num_subtasks, **kwargs)
        self.world_size = world_size
        self.world_shape = (len(self.world_objects), self.world_size, self.world_size)

        self.action_space = spaces.MultiDiscrete(
            np.array([num_subtasks + 1, 2 * self.n_lines, 2, 2])
        )
        self.observation_space.spaces.update(
            obs=spaces.Box(low=0, high=1, shape=self.world_shape),
            lines=spaces.MultiDiscrete(
                np.array(
                    [
                        [
                            len(self.line_types),
                            1 + len(self.interactions),
                            1 + len(self.objects),
                        ]
                    ]
                    * self.n_lines
                )
            ),
        )

    def line_str(self, line: Line):
        if isinstance(line, Subtask):
            i, o = line.id
            return f"{line}: {i} {o}"
        elif isinstance(line, (If, While)):
            return f"{line}: {line.id}"
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
        if line is Padding:
            return [self.line_types.index(Padding), 0, 0]
        elif type(line) is Else:
            return [self.line_types.index(Else), 0, 0]
        elif type(line) is Subtask:
            i, o = line.id
            i, o = self.interactions.index(i), self.objects.index(o)
            return [self.line_types.index(Subtask), i + 1, o + 1]
        else:
            o = self.objects.index(line.id)
            return [self.line_types.index(type(line)), 0, o + 1]

    def state_generator(self, lines) -> State:
        assert self.max_nesting_depth == 1
        agent_pos = self.random.randint(0, self.world_size, size=2)
        offset = self.random.randint(1 + self.world_size - self.world_size, size=2)

        def world_array():
            world = np.zeros(self.world_shape)
            for o, p in object_pos + [("agent", agent_pos)]:
                p = np.array(p) + offset
                world[tuple((self.world_objects.index(o), *p))] = 1
            return world

        object_pos = self.populate_world(lines)
        line_iterator = self.line_generator(lines)
        condition_evaluations = defaultdict(list)
        times = Counter(on_subtask=0, to_complete=0)

        def evaluate_line(l):
            if l is None:
                return None
            line = lines[l]
            if type(line) is Subtask:
                return 1
            else:
                evaluation = any(o == line.id for o, _ in object_pos)
                if type(line) in (If, While):
                    condition_evaluations[type(line)] += [evaluation]
                return evaluation

        def get_nearest(to):
            candidates = [np.array(p) for o, p in object_pos if o == to]
            if candidates:
                return min(candidates, key=lambda k: np.sum(np.abs(agent_pos - k)))

        def next_subtask(l):
            l = line_iterator.send(evaluate_line(l))
            while not (l is None or type(lines[l]) is Subtask):
                l = line_iterator.send(evaluate_line(l))
            if l is not None:
                assert type(lines[l]) is Subtask
                _, o = lines[l].id
                n = get_nearest(o)
                if n is not None:
                    times["to_complete"] = 1 + np.max(np.abs(agent_pos - n))
                    times["on_subtask"] = 0
            return l

        possible_objects = [o for o, _ in object_pos]
        prev, curr = 0, next_subtask(None)
        term = False
        while True:
            term |= times["on_subtask"] - times["to_complete"] > self.time_to_waste
            subtask_id = yield State(
                obs=world_array(),
                condition=None,
                prev=prev,
                curr=curr,
                condition_evaluations=condition_evaluations,
                term=term,
            )
            times["on_subtask"] += 1
            interaction, obj = self.subtask_id_to_strings[subtask_id]

            def pair():
                return obj, tuple(agent_pos)

            def on_object():
                return pair() in object_pos  # standing on the desired object

            correct_id = lines[curr].id == (
                (interaction, obj) if type(lines[curr]) is Subtask else obj
            )
            if on_object():
                if interaction in (self.mine, self.bridge):
                    object_pos.remove(pair())
                    if correct_id:
                        possible_objects.remove(obj)
                    else:
                        term = True
                if interaction == self.bridge:
                    object_pos.append(("ice", tuple(agent_pos)))
                if correct_id:
                    prev, curr = curr, next_subtask(curr)
            else:
                nearest = get_nearest(obj)
                if nearest is not None:
                    agent_pos += np.clip(nearest - agent_pos, -1, 1)
                elif correct_id and obj not in possible_objects:
                    # subtask is impossible
                    prev, curr = curr, None

    def populate_world(self, lines):
        # place subtask objects
        line_io = [line.id for line in lines if type(line) is Subtask]
        line_pos = self.random.randint(0, self.world_size, size=(len(line_io), 2))
        object_pos = [
            (o, tuple(pos))
            for (interaction, o), pos in zip(line_io, line_pos)
            if o != "water"
        ]

        # prevent infinite loops
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
            obj = lines[while_line].id
            l = self.random.choice(block)
            i = self.random.choice(2)
            assert self.interactions[i] in (self.mine, self.bridge)
            line_id = self.interactions[i], obj
            lines[l] = Subtask(line_id)
            if not self.evaluating and obj in self.world_objects:
                num_obj = self.random.randint(self.max_while_objects + 1)
                if num_obj:
                    pos = self.random.randint(0, self.world_size, size=(num_obj, 2))
                    object_pos += [(obj, tuple(p)) for p in pos]
        return object_pos

    def assign_line_ids(self, lines):
        num_objects = len(self.objects)
        excluded = self.random.randint(num_objects, size=self.num_excluded_objects)
        included_objects = [o for i, o in enumerate(self.objects) if i not in excluded]

        interaction_ids = self.random.choice(len(self.interactions), size=len(lines))
        object_ids = self.random.choice(len(included_objects), size=len(lines))
        line_ids = self.random.choice(len(self.objects), size=len(lines))

        for line, line_id, interaction_id, object_id in zip(
            lines, line_ids, interaction_ids, object_ids
        ):
            if line is Subtask:
                subtask_id = (
                    self.interactions[interaction_id],
                    included_objects[object_id],
                )
                yield Subtask(subtask_id)
            else:
                yield line(self.objects[line_id])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = build_parser(parser)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(Env(**args, baseline=False), action_fn=action_fn)
