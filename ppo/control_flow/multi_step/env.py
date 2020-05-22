import functools
import itertools
from collections import Counter, namedtuple, deque
from copy import deepcopy
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
        max_while_loops,
        use_water,
        max_failure_sample_prob,
        one_condition,
        failure_buffer_size,
        world_size=6,
        **kwargs,
    ):
        self.one_condition = one_condition
        self.max_failure_sample_prob = max_failure_sample_prob
        self.failure_buffer = deque(maxlen=failure_buffer_size)
        self.max_world_resamples = max_world_resamples
        self.max_while_loops = max_while_loops
        self.term_on = term_on
        self.temporal_extension = temporal_extension
        self.loops = None
        self.whiles = None
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
                    upper=num_subtasks + 1,
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
        print(self.i)
        print(inventory)
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

    def line_str(self, line):
        line = super().line_str(line)
        if type(line) is Subtask:
            return f"{line} {self.subtasks.index(line.id)}"
        elif type(line) in (If, While):
            if self.one_condition:
                evaluation = "counts[iron] > counts[gold]"
            elif line.id == Env.iron:
                evaluation = "counts[iron] > counts[gold]"
            elif line.id == Env.gold:
                evaluation = "counts[gold] > counts[merchant]"
            elif line.id == Env.wood:
                evaluation = "counts[merchant] > counts[iron]"
            return f"{line} {evaluation}"
        return line

    @staticmethod
    @functools.lru_cache(maxsize=200)
    def preprocess_line(line, **kwargs):
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

    def evaluate_line(self, line, counts, condition_evaluations, loops):
        if line is None:
            return None
        elif type(line) is Loop:
            return loops > 0
        elif type(line) in (If, While):
            if self.one_condition:
                evaluation = counts[Env.iron] > counts[Env.gold]
            elif line.id == Env.iron:
                evaluation = counts[Env.iron] > counts[Env.gold]
            elif line.id == Env.gold:
                evaluation = counts[Env.gold] > counts[Env.merchant]
            elif line.id == Env.wood:
                evaluation = counts[Env.merchant] > counts[Env.iron]
            else:
                raise RuntimeError
            condition_evaluations += [evaluation]
            return evaluation

    def feasible(self, objects, lines):
        line_iterator = self.line_generator(lines)
        l = next(line_iterator)
        loops = 0
        whiles = 0
        inventory = Counter()
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
                for r in required:
                    if counts[r] <= (1 if r == self.wood else 0):
                        return False
                if behavior in self.sell:
                    if inventory[resource] == 0:
                        # collect from environment
                        counts[resource] -= 1
                        inventory[resource] += 1
                    inventory[resource] -= 1
                elif behavior == self.mine:
                    counts[resource] -= 1
                    inventory[resource] += 1
            elif type(line) is Loop:
                loops += 1
            elif type(line) is While:
                whiles += 1
                if whiles > self.max_while_loops:
                    return False
            evaluation = self.evaluate_line(line, counts, [], loops)
            l = line_iterator.send(evaluation)
        return True

    @staticmethod
    def count_objects(objects):
        counts = Counter()
        for o in objects.values():
            counts[o] += 1
        return counts

    def generators(self) -> Tuple[Iterator[State], List[Line]]:
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
            lines, objects, _agent_pos = self.failure_buffer[choice]
            del self.failure_buffer[choice]
        else:
            while True:
                n_lines = (
                    self.random.random_integers(
                        self.min_eval_lines, self.max_eval_lines
                    )
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
                line_types = [While, Subtask, EndWhile, Subtask]
                lines = list(self.assign_line_ids(line_types))
                assert self.max_nesting_depth == 1
                result = self.populate_world(lines)
                if result is not None:
                    _agent_pos, objects = result
                    break

        def state_generator(agent_pos) -> State:
            initial_objects = deepcopy(objects)
            initial_agent_pos = deepcopy(agent_pos)
            line_iterator = self.line_generator(lines)
            condition_evaluations = []
            if self.lower_level == "train-alone":
                self.time_remaining = 0
            else:
                self.time_remaining = 200 if self.evaluating else self.time_to_waste
            self.loops = None
            self.whiles = 0
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
                        e = self.evaluate_line(
                            lines[l], counts, condition_evaluations, self.loops
                        )
                        # if e:
                        # import ipdb

                        # ipdb.set_trace()
                        l = line_iterator.send(e)
                        if self.loops == 0:
                            self.loops = None
                    if l is None or type(lines[l]) is Subtask:
                        break
                if l is not None:
                    assert type(lines[l]) is Subtask
                    time_delta = 3 * self.world_size
                    if self.lower_level == "train-alone":
                        self.time_remaining = time_delta + self.time_to_waste
                    else:
                        self.time_remaining += time_delta
                    return l

            prev, ptr = 0, next_subtask(None)
            term = False
            while True:
                term |= not self.time_remaining
                if term and ptr is not None:
                    self.failure_buffer.append(
                        (lines, initial_objects, initial_agent_pos)
                    )

                subtask_id, lower_level_index = yield State(
                    obs=(self.world_array(objects, agent_pos), inventory),
                    prev=prev,
                    ptr=ptr,
                    term=term,
                    subtask_complete=subtask_complete,
                    use_failure_buf=use_failure_buf,
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
                # interaction, obj = lines[agent_ptr].id
                interaction, resource = self.subtasks[subtask_id]

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
                if tgt_obj not in objects.values() and (
                    tgt_interaction != Env.sell or inventory[tgt_obj] == 0
                ):
                    term = True

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
                                pass  # TODO
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

        return state_generator(_agent_pos), lines

    def populate_world(self, lines):
        feasible = False
        use_water = False
        max_random_objects = 0
        object_list = []
        for i in range(self.max_world_resamples):
            max_random_objects = self.world_size ** 2
            num_random_objects = np.random.randint(max_random_objects)
            object_list = [self.agent] + list(
                self.random.choice(
                    self.items + [self.merchant], size=num_random_objects
                )
            )
            feasible = self.feasible(object_list, lines)

            if feasible:
                use_water = (
                    self.use_water
                    and num_random_objects < max_random_objects - self.world_size
                )
                break

        if not feasible:
            return None

        if use_water:
            vertical_water = self.random.choice(2)
            world_shape = (
                [self.world_size, self.world_size - 1]
                if vertical_water
                else [self.world_size - 1, self.world_size]
            )
        else:
            world_shape = (self.world_size, self.world_size)
        indexes = self.random.choice(
            np.prod(world_shape),
            size=min(np.prod(world_shape), max_random_objects),
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
        if use_water:
            water_index = self.random.randint(1, self.world_size - 1)
            positions[positions[:, vertical_water] >= water_index] += np.array(
                [0, 1] if vertical_water else [1, 0]
            )
            assert water_index not in positions[:, vertical_water]
        objects = {
            tuple(p): (self.wall if o is None else o)
            for o, p in itertools.zip_longest(object_list, positions)
        }
        agent_pos = next(p for p, o in objects.items() if o == self.agent)
        del objects[agent_pos]
        if use_water:
            assert object_list[0] == self.agent
            agent_i, agent_j = positions[0]
            for p, o in objects.items():
                if o == self.wood:
                    pi, pj = p
                    if vertical_water:
                        if (water_index < pj and water_index < agent_j) or (
                            water_index > pj and water_index > agent_j
                        ):
                            objects = {
                                **objects,
                                **{
                                    (i, water_index): self.water
                                    for i in range(self.world_size)
                                },
                            }
                    else:
                        if (water_index < pi and water_index < agent_i) or (
                            water_index > pi and water_index > agent_i
                        ):
                            objects = {
                                **objects,
                                **{
                                    (water_index, i): self.water
                                    for i in range(self.world_size)
                                },
                            }

        return agent_pos, objects

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
    p, default_max_world_resamples=None, default_max_while_loops=None, **kwargs
):
    ppo.control_flow.env.build_parser(p, **kwargs)
    p.add_argument(
        "--no-temporal-extension", dest="temporal_extension", action="store_false"
    )
    p.add_argument("--no-water", dest="use_water", action="store_false")
    p.add_argument("--1condition", dest="one_condition", action="store_true")
    p.add_argument("--max-failure-sample-prob", type=float, required=True)
    p.add_argument("--failure-buffer-size", type=int, required=True)
    p.add_argument(
        "--max-world-resamples",
        type=int,
        required=default_max_world_resamples is None,
        default=default_max_world_resamples,
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
