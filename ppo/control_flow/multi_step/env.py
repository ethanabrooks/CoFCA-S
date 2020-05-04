import functools
import itertools
from collections import Counter, namedtuple
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
from ppo.control_flow.multi_step.rooms import rooms

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

Obs = namedtuple("Obs", "active lines obs inventory")


def get_nearest(_from, _to, objects):
    items = [(np.array(p), o) for p, o in objects.items()]
    candidates = [(p, np.max(np.abs(_from - p))) for p, o in items if o == _to]
    if candidates:
        return min(candidates, key=lambda c: c[1])


def objective(interaction, obj):
    if interaction == Env.sell:
        return Env.merchant
    return obj


def subtasks():
    for obj in Env.items:
        for interaction in Env.behaviors:
            yield interaction, obj


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
    colors = {
        wood: GREEN,
        gold: YELLOW,
        iron: LIGHTGREY,
        merchant: PINK,
        wall: RESET,
        water: BLUE,
        bridge: RESET,
        agent: RED,
    }

    def __init__(
        self,
        num_subtasks,
        temporal_extension,
        term_on,
        max_world_resamples,
        max_instruction_resamples,
        max_while_loops,
        use_water,
        world_size=6,
        **kwargs,
    ):
        self.max_instruction_resamples = max_instruction_resamples
        self.max_world_resamples = max_world_resamples
        self.max_while_loops = max_while_loops
        self.term_on = term_on
        self.temporal_extension = temporal_extension
        self.loops = None
        self.whiles = None
        self.impossible = None
        self.use_water = use_water

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
                ppo.control_flow.env.Action(
                    upper=num_subtasks + 2,
                    delta=2 * self.n_lines,
                    dg=2,
                    lower=len(self.lower_level_actions),
                    ptr=self.n_lines,
                )
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
            inventory=spaces.MultiBinary(len(self.items)),
        )

    def print_obs(self, obs):
        obs, inventory = obs
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
                    colors.append(self.colors[self.world_contents[x - 1]])
                    string.append(chars[x])
                colors.append(RESET)
                string.append("|")
            print(*[c for p in zip(colors, string) for c in p], sep="")
            print("-" * len(string))
        for i, c in zip(self.items, inventory):
            print(i, c)

    def line_str(self, line):
        line = super().line_str(line)
        if type(line) is Subtask:
            return f"{line} {self.subtasks.index(line.id)}"
        return line

    @staticmethod
    @functools.lru_cache(maxsize=200)
    def preprocess_line(line):
        def item_index(item):
            if item == Env.water:
                return len(Env.items)
            else:
                return Env.items.index(item)

        if type(line) in (Else, EndIf, EndWhile, EndLoop, Padding):
            return [Line.types.index(type(line)), 0, 0, 0]
        elif type(line) is Loop:
            return [Line.types.index(Loop), 0, 0, line.id]
        elif type(line) is Subtask:
            i, o = line.id
            i, o = Env.behaviors.index(i), item_index(o)
            return [Line.types.index(Subtask), i + 1, o + 1, 0]
        elif type(line) in (While, If):
            return [Line.types.index(type(line)), 0, item_index(line.id) + 1, 0]
        else:
            raise RuntimeError()

    def world_array(self, objects, agent_pos):
        world = np.zeros(self.world_shape)
        for p, o in list(objects.items()) + [(agent_pos, self.agent)]:
            p = np.array(p)
            world[tuple((self.world_contents.index(o), *p))] = 1

        return world

    @staticmethod
    def evaluate_line(line, counts, condition_evaluations, loops):
        if line is None:
            return None
        elif type(line) is Loop:
            return loops > 0
        elif type(line) in (If, While):
            if line.id == Env.iron:
                evaluation = counts[Env.iron] > counts[Env.gold]
            elif line.id == Env.gold:
                evaluation = counts[Env.gold] > counts[Env.iron]
            else:
                raise RuntimeError
            condition_evaluations += [evaluation]
            return evaluation
        else:
            return 1

    def feasible(self, objects, lines):
        line_iterator = self.line_generator(lines)
        l = next(line_iterator)
        loops = 0
        whiles = 0
        counts = Counter()
        for o in objects:
            counts[o] += 1
        while l is not None:
            line = lines[l]
            if type(line) is Subtask:
                behavior, resource = line.id
                if behavior == self.sell:
                    required = {self.merchant, resource}
                elif behavior == self.mine:
                    required = {resource}
                else:
                    required = {resource}
                for resource in required:
                    if counts[resource] <= 0:
                        return False
                if behavior in self.sell:
                    counts[resource] -= 1
                elif behavior == self.mine:
                    counts[resource] -= 1
            elif type(line) is Loop:
                loops += 1
            elif type(line) is While:
                whiles += 1
                if whiles > self.max_while_loops:
                    return True
            evaluation = self.evaluate_line(line, counts, [], loops)
            l = line_iterator.send(evaluation)
        return True

    @staticmethod
    def count_objects(objects):
        counts = Counter()
        for o in objects.values():
            counts[o] += 1
        return counts

    def generators(self, count=0) -> Tuple[Iterator[State], List[Line]]:
        n_lines = (
            self.eval_lines
            if self.evaluating
            else self.random.random_integers(self.min_lines, self.max_lines)
        )
        line_types = list(
            Line.generate_types(
                n_lines,
                remaining_depth=self.max_nesting_depth,
                random=self.random,
                legal_lines=self.control_flow_types,
            )
        )
        lines = list(self.assign_line_ids(line_types))
        assert self.max_nesting_depth == 1
        agent_pos, objects, feasible = self.populate_world(lines)
        if not feasible and count < self.max_instruction_resamples:
            return self.generators(count + 1)

        def state_generator(agent_pos) -> State:
            line_iterator = self.line_generator(lines)
            condition_evaluations = []
            if self.lower_level == "train-alone":
                self.time_remaining = 0
            else:
                self.time_remaining = 200 if self.evaluating else self.time_to_waste
            self.loops = None
            self.whiles = 0
            self.impossible = False
            inventory = Counter()
            subtask_complete = False

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
                        elif type(lines[l]) is While:
                            self.whiles += 1
                            if self.whiles > self.max_while_loops:
                                return None
                        counts = self.count_objects(objects)
                        l = line_iterator.send(
                            self.evaluate_line(
                                lines[l], counts, condition_evaluations, self.loops
                            )
                        )
                        if self.loops == 0:
                            self.loops = None
                    if l is None or type(lines[l]) is Subtask:
                        break
                if l is not None:
                    assert type(lines[l]) is Subtask
                    be, it = lines[l].id
                    if it not in objects.values():
                        self.impossible = True
                    elif be == self.sell:
                        if self.merchant not in objects.values():
                            self.impossible = True
                    time_delta = 3 * self.world_size
                    if self.lower_level == "train-alone":
                        self.time_remaining = time_delta + self.time_to_waste
                    else:
                        self.time_remaining += time_delta
                    return l

            possible_objects = list(objects.values())
            prev, ptr = 0, next_subtask(None)
            term = False
            while True:
                term |= not self.time_remaining
                subtask_id, lower_level_index = yield State(
                    obs=(self.world_array(objects, agent_pos), inventory),
                    prev=prev,
                    ptr=ptr,
                    term=term,
                    subtask_complete=subtask_complete,
                    impossible=self.impossible,
                )
                subtask_complete = False
                # for i, a in enumerate(self.lower_level_actions):
                # print(i, a)
                # try:
                # lower_level_index = int(input("go:"))
                # except ValueError:
                # pass
                if self.lower_level == "train-alone":
                    interaction, resource = lines[ptr].id
                elif subtask_id < len(self.subtasks):
                    # interaction, obj = lines[agent_ptr].id
                    interaction, resource = self.subtasks[subtask_id]
                else:
                    if self.impossible:
                        ptr = None  # success
                    else:
                        term = True  # failure
                    yield State(
                        obs=(self.world_array(objects, agent_pos), inventory),
                        prev=prev,
                        ptr=ptr,
                        term=term,
                        subtask_complete=subtask_complete,
                        impossible=self.impossible,
                    )

                if self.lower_level == "hardcoded":
                    lower_level_action = self.get_lower_level_action(
                        interaction=interaction,
                        resource=resource,
                        agent_pos=agent_pos,
                        objects=objects,
                    )
                    # print("lower level action:", lower_level_action)
                else:
                    lower_level_action = self.lower_level_actions[lower_level_index]
                self.time_remaining -= 1
                tgt_interaction, tgt_obj = lines[ptr].id

                if type(lower_level_action) is str:
                    standing_on = objects.get(tuple(agent_pos), None)
                    done = (
                        lower_level_action == tgt_interaction
                        and standing_on == objective(*lines[ptr].id)
                    )
                    if lower_level_action == self.mine:
                        if tuple(agent_pos) in objects:
                            if (
                                done
                                or (
                                    tgt_interaction == self.sell
                                    and standing_on == tgt_obj
                                )
                                or standing_on == self.wood
                            ):
                                if While in self.control_flow_types:
                                    possible_objects.remove(standing_on)
                            elif self.mine in self.term_on:
                                term = True
                            if (
                                standing_on in self.items
                                and inventory[standing_on] == 0
                            ):
                                inventory[standing_on] = 1
                            del objects[tuple(agent_pos)]
                    elif lower_level_action == self.sell:
                        done = done and (
                            self.lower_level == "hardcoded" or inventory[tgt_obj] > 0
                        )
                        if done:
                            inventory[tgt_obj] -= 1
                        elif self.sell in self.term_on:
                            term = True
                    elif (
                        lower_level_action == self.goto
                        and not done
                        and self.goto in self.term_on
                    ):
                        term = True
                    if done:
                        prev, ptr = ptr, next_subtask(ptr)
                        subtask_complete = True

                elif type(lower_level_action) is np.ndarray:
                    if self.temporal_extension:
                        lower_level_action = np.clip(lower_level_action, -1, 1)
                    new_pos = agent_pos + lower_level_action
                    moving_into = objects.get(tuple(new_pos), None)
                    if (
                        np.all(0 <= new_pos)
                        and np.all(new_pos < self.world_size)
                        and (
                            self.lower_level == "hardcoded"
                            or (
                                moving_into != self.wall
                                and (
                                    moving_into != self.water
                                    or inventory[self.wood] > 0
                                )
                            )
                        )
                    ):
                        agent_pos = new_pos
                        if moving_into == self.water:
                            # build bridge
                            del objects[tuple(new_pos)]
                            inventory[self.wood] -= 1
                else:
                    assert lower_level_action is None

        return state_generator(agent_pos), lines

    def populate_world(self, lines, count=0):
        room = np.array([list(r) for r in self.random.choice(rooms).split("\n")])
        unoccupied = room == " "
        max_random_objects = unoccupied.sum()
        random_agent = self.agent[0] not in room
        num_random_objects = np.random.randint(
            1 if random_agent else 0, max_random_objects
        )
        object_list = list(
            self.random.choice(
                self.items + [self.merchant],
                size=num_random_objects - 1 if random_agent else num_random_objects,
            )
        )
        feasible = True
        if not self.feasible(object_list, lines):
            if count >= self.max_world_resamples:
                feasible = False
            else:
                return self.populate_world(lines, count=count + 1)

        ij = np.stack(
            np.meshgrid(np.arange(self.world_size), np.arange(self.world_size)), axis=-1
        ).swapaxes(0, 1)
        unoccupied = ij[unoccupied]
        object_ij = list(
            unoccupied[
                self.random.choice(
                    len(unoccupied), size=num_random_objects, replace=False
                )
            ]
        )
        if random_agent:
            assert self.agent[0] not in room
            agent_pos = object_ij.pop()
        else:
            assert self.agent[0] in room
            agent_pos = ij[room == self.agent[0]][0]

        occupied = room != " "
        positions = object_ij + list(ij[occupied])
        world_contents = {t[0]: t for t in self.world_contents}
        object_list += [world_contents[x] for x in room[occupied]]
        objects = {tuple(p): o for p, o in (zip(positions, object_list))}
        return agent_pos, objects, feasible

    def assign_line_ids(self, line_types):
        behaviors = self.random.choice(self.behaviors, size=len(line_types))
        items = self.random.choice([self.gold, self.iron], size=len(line_types))
        while_obj = None
        available = [x for x in self.items]
        lines = []
        for line_type, behavior, item in zip(line_types, behaviors, items):
            if line_type is Subtask:
                if not available:
                    return self.assign_line_ids(line_types)
                subtask_id = (behavior, self.random.choice(available))
                lines += [Subtask(subtask_id)]
            elif line_type is Loop:
                lines += [Loop(self.random.randint(1, 1 + self.max_loops))]
            elif line_type is While:
                while_obj = item
                lines += [line_type(item)]
            elif line_type is If:
                lines += [line_type(item)]
            elif line_type is EndWhile:
                if while_obj in available:
                    available.remove(while_obj)
                while_obj = None
                lines += [EndWhile(0)]
            else:
                lines += [line_type(0)]
        return lines

    def get_observation(self, obs, **kwargs):
        obs, inventory = obs
        obs = super().get_observation(obs=obs, **kwargs)
        obs.update(inventory=np.array([inventory[i] for i in self.items]))
        # if not self.observation_space.contains(obs):
        #     import ipdb
        #
        #     ipdb.set_trace()
        #     self.observation_space.contains(obs)
        return obs

    @staticmethod
    def get_lower_level_action(interaction, resource, agent_pos, objects):
        resource = objective(interaction, resource)
        if objects.get(tuple(agent_pos), None) == resource:
            return interaction
        else:
            nearest = get_nearest(_from=agent_pos, _to=resource, objects=objects)
            if nearest:
                n, d = nearest
                return n - agent_pos


def build_parser(
    p,
    default_max_world_resamples=None,
    default_max_while_loops=None,
    default_max_instruction_resamples=None,
    **kwargs,
):
    ppo.control_flow.env.build_parser(p, **kwargs)
    p.add_argument(
        "--no-temporal-extension", dest="temporal_extension", action="store_false"
    )
    p.add_argument("--no-water", dest="use_water", action="store_false")
    p.add_argument(
        "--max-world-resamples",
        type=int,
        required=default_max_world_resamples is None,
        default=default_max_world_resamples,
    )
    p.add_argument(
        "--max-instruction-resamples",
        type=int,
        required=default_max_instruction_resamples is None,
        default=default_max_instruction_resamples,
    )
    p.add_argument(
        "--max-while-loops",
        type=int,
        required=default_max_while_loops is None,
        default=default_max_while_loops,
    )
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument(
        "--term-on", nargs="+", choices=[Env.sell, Env.mine, Env.goto], required=True
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    build_parser(parser)
    parser.add_argument("--seed", default=0, type=int)
    ppo.control_flow.env.main(
        Env(rank=0, lower_level="train-alone", **hierarchical_parse_args(parser))
    )
