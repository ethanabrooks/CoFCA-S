from collections import Counter, defaultdict
from copy import copy

import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo import keyboard_control
from ppo.control_flow.env import build_parser, State
from ppo.control_flow.lines import Subtask, Padding, Line, While, If, EndWhile


class Env(ppo.control_flow.env.Env):
    subtask_objects = ["pig", "sheep", "cat", "greenbot"]
    other_objects = ["ice", "agent"]
    line_objects = subtask_objects + ["monkey"]
    world_objects = subtask_objects + other_objects
    interactions = ["pickup", "transform", "visit"]

    def __init__(self, world_size, num_subtasks, add_while_obj_prob, **kwargs):
        num_subtasks = len(self.subtask_objects) * len(self.interactions)
        super().__init__(num_subtasks=num_subtasks, **kwargs)
        self.add_while_obj_prob = add_while_obj_prob
        self.world_size = world_size
        self.world_shape = (len(self.world_objects), self.world_size, self.world_size)
        self.action_space = spaces.MultiDiscrete(
            np.array([self.num_subtasks + 1, 2 * self.n_lines, 2, 2])
        )
        self.observation_space.spaces.update(
            obs=spaces.Box(low=0, high=1, shape=self.world_shape),
            lines=spaces.MultiDiscrete(
                np.array(
                    [
                        [
                            len(self.line_types),
                            1 + len(self.interactions),
                            1 + len(self.line_objects),
                        ]
                    ]
                    * self.n_lines
                )
            ),
        )

    def line_str(self, line: Line):
        i, o = self.parse_id(line.id)
        if isinstance(line, Subtask):
            return f"{line}: {i} {o}"
        elif isinstance(line, (If, While)):
            return f"{line}: {o}"
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

    def preprocess_line(self, line):
        if line is Padding:
            return [self.line_types.index(Padding), 0, 0]
        else:
            i, o = self.parse_id(line.id)
            line_type = self.line_types.index(type(line))
            return [
                line_type,
                1 + self.interactions.index(i) if type(line) is Subtask else 0,
                1 + self.line_objects.index(o),
            ]

    def state_generator(self, lines) -> State:
        assert self.max_nesting_depth == 1
        agent_pos = self.random.randint(0, self.world_size, size=2)

        def build_world():
            world = np.zeros(self.world_shape)
            for o, p in object_pos + [("agent", agent_pos)]:
                world[tuple((self.world_objects.index(o), *p))] = 1
            return world

        line_io = [self.parse_id(line.id) for line in lines if type(line) is Subtask]
        line_pos = self.random.randint(0, self.world_size, size=(len(line_io), 2))
        object_pos = [
            (o, tuple(pos)) for (interaction, o), pos in zip(line_io, line_pos)
        ]

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
            _, obj = self.parse_id(lines[while_line].id)
            l = self.random.choice(block)
            i = self.random.choice(2)
            assert self.interactions[i] in ("pickup", "transform")
            o = self.line_objects.index(obj)
            line_id = o * len(self.interactions) + i
            assert self.parse_id(line_id) in (("pickup", obj), ("transform", obj))
            lines[l] = Subtask(line_id)
            if (
                not self.evaluating
                and self.random.random() < self.add_while_obj_prob
                and obj in self.world_objects
            ):
                object_pos += [
                    (obj, tuple(self.random.randint(0, self.world_size, size=2)))
                ]

        line_iterator = self.line_generator(lines)
        condition_evaluations = defaultdict(list)

        def evaluate_line(l):
            if l is None:
                return None
            line = lines[l]
            if type(line) is Subtask:
                return 1
            else:
                _, tgt = self.parse_id(line.id)
                evaluation = any(o == tgt for o, _ in object_pos)
                if type(line) in (If, While):
                    condition_evaluations[type(line)] += [evaluation]
                return evaluation

        def next_subtask(l):
            l = line_iterator.send(evaluate_line(l))
            while not (l is None or type(lines[l]) is Subtask):
                l = line_iterator.send(evaluate_line(l))
            return l

        prev, curr = 0, next_subtask(None)
        while True:
            subtask_id = yield State(
                obs=build_world(),
                condition=None,
                prev=prev,
                curr=curr,
                condition_evaluations=condition_evaluations,
            )
            interaction, obj = self.parse_id(subtask_id)

            def pair():
                return obj, tuple(agent_pos)

            def on_object():
                return pair() in object_pos  # standing on the desired object

            correct_id = subtask_id == lines[curr].id
            if on_object():
                if interaction in ("pickup", "transform"):
                    object_pos.remove(pair())
                if interaction == "transform":
                    object_pos.append(("ice", tuple(agent_pos)))
                prev, curr = curr, next_subtask(curr)
            else:
                candidates = [np.array(p) for o, p in object_pos if o == obj]
                if candidates:
                    nearest = min(
                        candidates, key=lambda k: np.sum(np.abs(agent_pos - k))
                    )
                    agent_pos += np.clip(nearest - agent_pos, -1, 1)
                elif correct_id:
                    # subtask is impossible
                    prev, curr = curr, next_subtask(curr)

    def build_lines(self):
        num_line_ids = len(self.interactions) * len(self.line_objects)
        return [
            line(self.random.randint(num_line_ids))
            if type(line) not in (Subtask, Padding)
            else line
            for line in (super().build_lines())
        ]

    def parse_id(self, line_id):
        i = line_id % len(self.interactions)
        o = line_id // len(self.interactions)
        return self.interactions[i], self.line_objects[o]


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
