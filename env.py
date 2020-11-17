import functools
import itertools
import pickle
from collections import Counter, namedtuple, deque, OrderedDict, defaultdict
from copy import deepcopy

import gym
import numpy as np
import redis
from gym import spaces
from gym.utils import seeding
from utils import (
    hierarchical_parse_args,
    RESET,
    GREEN,
    RED,
    YELLOW,
    LIGHTGREY,
    PINK,
    BLUE,
)
from typing import List, Tuple, Dict, Optional, Generator

import keyboard_control
from lines import (
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

Coord = Tuple[int, int]
ObjectMap = Dict[Coord, str]

Obs = namedtuple("Obs", "active inventory lines mask obs subtask_complete truthy")
assert tuple(Obs._fields) == tuple(sorted(Obs._fields))

Last = namedtuple("Last", "action active reward terminal selected")
State = namedtuple(
    "State", "obs prev ptr term subtask_complete time_remaining counts inventory"
)
Action = namedtuple("Action", "upper lower delta dg ptr")


def objective(interaction, obj):
    if interaction == Env.sell:
        return Env.merchant
    return obj


def subtasks():
    for obj in Env.items:
        for interaction in Env.behaviors:
            yield interaction, obj


class Env(gym.Env):
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
        r: redis.Redis,
        max_world_resamples: int,
        max_while_loops: int,
        use_water: bool,
        max_failure_sample_prob: int,
        one_condition: bool,
        failure_buffer_size: int,
        reject_while_prob: float,
        long_jump: bool,
        min_eval_lines: int,
        max_eval_lines: int,
        min_lines: int,
        max_lines: int,
        eval_condition_size: int,
        single_control_flow_type: bool,
        no_op_limit: int,
        time_to_waste: int,
        subtasks_only: bool,
        break_on_fail: bool,
        max_loops: int,
        rank: int,
        lower_level: str,
        control_flow_types=None,
        evaluating=False,
        max_nesting_depth=1,
        seed=0,
        term_on=None,
        world_size=6,
    ):
        self.r = r
        if control_flow_types is None:
            control_flow_types = [Subtask, If, While, Else]
        if term_on is None:
            term_on = [self.mine, self.sell]
        self.counts = None
        self.reject_while_prob = reject_while_prob
        self.one_condition = one_condition
        self.max_failure_sample_prob = max_failure_sample_prob
        self.failure_buffer = deque(maxlen=failure_buffer_size)
        self.max_world_resamples = max_world_resamples
        self.max_while_loops = max_while_loops
        self.term_on = term_on
        self.use_water = use_water

        self.subtasks = list(subtasks())
        num_subtasks = len(self.subtasks)
        self.min_eval_lines = min_eval_lines
        self.max_eval_lines = max_eval_lines
        self.lower_level = lower_level
        if Subtask not in control_flow_types:
            control_flow_types.append(Subtask)
        self.control_flow_types = control_flow_types
        self.rank = rank
        self.max_loops = max_loops
        self.break_on_fail = break_on_fail
        self.subtasks_only = subtasks_only
        self.no_op_limit = no_op_limit
        self._eval_condition_size = eval_condition_size
        self.single_control_flow_type = single_control_flow_type
        self.max_nesting_depth = max_nesting_depth
        self.num_subtasks = num_subtasks
        self.time_to_waste = time_to_waste
        self.i = 0
        self.success_count = 0

        self.min_lines = min_lines
        self.max_lines = max_lines
        if evaluating:
            self.n_lines = max_eval_lines
        else:
            self.n_lines = max_lines
        self.n_lines += 1
        self.random, self.seed = seeding.np_random(seed)
        self.evaluating = evaluating
        self.iterator = None
        self._render = None

        def possible_lines():
            for i in range(num_subtasks):
                yield Subtask(i)
            for i in range(1, max_loops + 1):
                yield Loop(i)
            for line_type in self.line_types:
                if line_type not in (Subtask, Loop):
                    yield line_type(0)

        self.possible_lines = list(possible_lines())
        self.long_jump = long_jump and self.evaluating
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
                Action(
                    upper=num_subtasks + 1,
                    delta=2 * self.n_lines,
                    dg=2,
                    lower=len(self.lower_level_actions),
                    ptr=self.n_lines,
                )
            )
        )
        lines_space = spaces.MultiDiscrete(
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
        )
        mask_space = spaces.MultiDiscrete(2 * np.ones(self.n_lines))
        self.observation_space = spaces.Dict(
            Obs(
                active=spaces.Discrete(self.n_lines + 1),
                inventory=spaces.MultiBinary(len(self.items)),
                lines=lines_space,
                mask=mask_space,
                obs=spaces.Box(low=0, high=1, shape=self.world_shape, dtype=np.float32),
                subtask_complete=spaces.Discrete(2),
                truthy=spaces.MultiDiscrete(4 * np.ones(self.n_lines)),
            )._asdict()
        )
        self.world_space = spaces.Box(
            low=0, high=self.world_size - 1, shape=[2], dtype=np.float32
        )

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

    def evaluate_line(self, line: Line, loops: Optional[int], counts: Dict[str, int]):
        if line is None:
            return None
        elif type(line) is Loop:
            assert loops is not None
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
            return evaluation

    def feasible(self, objects, lines) -> bool:
        line_iterator = self.line_generator(lines)
        line = next(line_iterator)
        loops = 0
        whiles = 0
        inventory = Counter()
        counts = Counter()
        for o in objects:
            counts[o] += 1
        while line is not None:
            line = lines[line]
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
                if all(
                    (
                        whiles == 0,  # first loop
                        not self.evaluate_line(line, loops, counts),  # evaluates false
                        not self.evaluating,
                        self.random.random() < self.reject_while_prob,
                    )
                ):
                    return False
                whiles += 1
                if whiles > self.max_while_loops:
                    return False
            evaluation = self.evaluate_line(line, loops, counts)
            if evaluation is True and self.long_jump:
                assert self.evaluating
                return False
            line = line_iterator.send(evaluation)
        return True

    @staticmethod
    def count_objects(objects):
        counts = Counter()
        for o in objects.values():
            counts[o] += 1
        return counts

    def subtask_generator(self, line_iterator, lines, **kwargs):
        line = next(line_iterator)
        loops = None
        whiles = 0
        while True:
            if line is None:
                kwargs = yield None
                line = next(line_iterator)
            else:
                if type(lines[line]) is Subtask:
                    assert type(lines[line]) is Subtask
                    kwargs = yield line
                if type(lines[line]) is Loop:
                    if loops is None:
                        loops = lines[line].id
                    else:
                        loops -= 1
                elif type(lines[line]) is While:
                    whiles += 1
                    if whiles > self.max_while_loops:
                        kwargs = yield None

                line = line_iterator.send(
                    self.evaluate_line(lines[line], loops, **kwargs)
                )
                if loops == 0:
                    loops = None

    def state_generator(
        self, objects: ObjectMap, agent_pos: Coord, lines: List[Line]
    ) -> Generator[State, Tuple[int, int], None]:
        initial_objects = deepcopy(objects)
        initial_agent_pos = deepcopy(agent_pos)
        line_iterator = self.line_generator(lines)
        if self.lower_level == "train-alone":
            time_remaining = 0
        else:
            time_remaining = 200 if self.evaluating else self.time_to_waste
        inventory = Counter()
        subtask_complete = False
        subtask_iterator = self.subtask_generator(
            line_iterator, lines, counts=self.count_objects(objects)
        )

        def update_time():
            time_delta = 3 * self.world_size
            return (
                time_delta + self.time_to_waste
                if self.lower_level == "train-alone"
                else time_remaining + time_delta
            )

        prev, ptr = 0, next(subtask_iterator)
        if ptr is not None:
            time_remaining = update_time()

        term = False
        while True:
            term |= not time_remaining
            if term and ptr is not None:
                self.failure_buffer.append((lines, initial_objects, initial_agent_pos))

            def state(terminate):

                world = np.zeros(self.world_shape)
                for p, o in list(objects.items()) + [(agent_pos, self.agent)]:
                    p = np.array(p)
                    world[tuple((self.world_contents.index(o), *p))] = 1

                return State(
                    obs=world,
                    prev=prev,
                    ptr=ptr,
                    term=terminate,
                    subtask_complete=subtask_complete,
                    time_remaining=time_remaining,
                    counts=self.count_objects(objects),
                    inventory=inventory,
                )

            # noinspection PyTupleAssignmentBalance
            subtask_id, lower_level_index = yield state(term)
            subtask_complete = False
            # for i, a in enumerate(self.lower_level_actions):
            # print(i, a)
            # try:
            # lower_level_index = int(input("go:"))
            # except ValueError:
            # pass

            lower_level_action = self.lower_level_actions[lower_level_index]
            time_remaining -= 1
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
                        good_mine = (
                            done
                            or (tgt_interaction == self.sell and standing_on == tgt_obj)
                            or standing_on == self.wood
                        )
                        if not good_mine and self.mine in self.term_on:
                            term = True
                        if standing_on in self.items and inventory[standing_on] == 0:
                            inventory[standing_on] = 1
                        del objects[tuple(agent_pos)]
                elif lower_level_action == self.sell:
                    done = done and (inventory[tgt_obj] > 0)
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
                    prev, ptr = (
                        ptr,
                        subtask_iterator.send(dict(counts=self.count_objects(objects))),
                    )
                    if ptr is not None:
                        time_remaining = update_time()
                    subtask_complete = True

            elif type(lower_level_action) is np.ndarray:
                lower_level_action = np.clip(lower_level_action, -1, 1)
                new_pos = agent_pos + lower_level_action
                moving_into = objects.get(tuple(new_pos), None)
                if self.world_space.contains(new_pos) and (
                    moving_into != self.wall
                    and (moving_into != self.water or inventory[self.wood] > 0)
                ):
                    agent_pos = new_pos
                    if moving_into == self.water:
                        # build bridge
                        del objects[tuple(new_pos)]
                        inventory[self.wood] -= 1
            else:
                assert lower_level_action is None

    def populate_world(self, lines) -> Optional[Tuple[Coord, ObjectMap]]:
        feasible = False
        use_water = False
        max_random_objects = 0
        object_list = []
        for i in range(self.max_world_resamples):
            max_random_objects = self.world_size ** 2
            num_random_objects = self.random.randint(max_random_objects)
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
            vertical_water = False
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
        water_index = 0
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
        items = self.random.choice(self.items, size=len(line_types))
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

    @property
    def line_types(self):
        return [If, Else, EndIf, While, EndWhile, EndLoop, Subtask, Padding, Loop]
        # return list(Line.types)

    def reset(self):
        self.i += 1
        self.iterator = self.generator()
        s, r, t, i = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def render_world(
        self,
        state,
        action,
        lower_level_action,
        reward,
    ):

        if action is not None and action < len(self.subtasks):
            print("Selected:", self.subtasks[action], action)
        print("Action:", action)
        if lower_level_action is not None:
            print(
                "Lower Level Action:",
                self.lower_level_actions[lower_level_action],
            )
        print("Reward", reward)
        print("Time remaining", state.time_remaining)
        print("Obs:")
        _obs = state.obs
        _obs = _obs.transpose(1, 2, 0).astype(int)
        grid_size = 3  # obs.astype(int).sum(-1).max()  # max objects per grid
        chars = [" "] + [o for (o, *_) in self.world_contents]
        print(self.i)
        print(state.inventory)
        for i, row in enumerate(_obs):
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

    def render_instruction(
        self,
        term,
        success,
        lines,
        state,
        agent_ptr,
    ):

        if term:
            print(GREEN if success else RED)
        indent = 0
        for i, line in enumerate(lines):
            if i == state.ptr and i == agent_ptr:
                pre = "+ "
            elif i == agent_ptr:
                pre = "- "
            elif i == state.ptr:
                pre = "| "
            else:
                pre = "  "
            indent += line.depth_change[0]
            if type(line) in (If, While):
                if self.one_condition:
                    evaluation = f"counts[iron] ({state.counts[self.iron]}) > counts[gold] ({state.counts[self.gold]})"
                elif line.id == Env.iron:
                    evaluation = f"counts[iron] ({state.counts[self.iron]}) > counts[gold] ({state.counts[self.gold]})"
                elif line.id == Env.gold:
                    evaluation = f"counts[gold] ({state.counts[self.gold]}) > counts[merchant] ({state.counts[self.merchant]})"
                elif line.id == Env.wood:
                    evaluation = f"counts[merchant] ({state.counts[self.merchant]}) > counts[iron] ({state.counts[self.iron]})"
                else:
                    raise RuntimeError
                line_str = f"{line} {evaluation}"
            else:
                line_str = str(line)
            print("{:2}{}{}{}".format(i, pre, " " * indent, line_str))
            indent += line.depth_change[1]
        print(RESET)

    def generator(self):
        step = 0
        n = 0
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
                if self.long_jump:
                    assert self.evaluating
                    len_jump = self.random.randint(
                        self.min_eval_lines - 3, self.max_eval_lines - 3
                    )
                    use_if = self.random.random() < 0.5
                    line_types = [
                        If if use_if else While,
                        *(Subtask for _ in range(len_jump)),
                        EndIf if use_if else EndWhile,
                        Subtask,
                    ]
                elif self.single_control_flow_type and self.evaluating:
                    assert n_lines >= 6
                    while True:
                        line_types = list(
                            Line.generate_types(
                                n_lines,
                                remaining_depth=self.max_nesting_depth,
                                random=self.random,
                                legal_lines=self.control_flow_types,
                            )
                        )
                        criteria = [
                            Else in line_types,  # Else
                            While in line_types,  # While
                            line_types.count(If) > line_types.count(Else),  # If
                        ]
                        if sum(criteria) >= 2:
                            break
                else:
                    legal_lines = (
                        [
                            self.random.choice(
                                list(set(self.control_flow_types) - {Subtask})
                            ),
                            Subtask,
                        ]
                        if (self.single_control_flow_type and not self.evaluating)
                        else self.control_flow_types
                    )

                    line_types = list(
                        Line.generate_types(
                            n_lines,
                            remaining_depth=self.max_nesting_depth,
                            random=self.random,
                            legal_lines=legal_lines,
                        )
                    )
                lines = list(self.assign_line_ids(line_types))
                assert self.max_nesting_depth == 1
                result = self.populate_world(lines)
                if result is not None:
                    _agent_pos, objects = result
                    break

        state_iterator = self.state_generator(objects, _agent_pos, lines)
        state = next(state_iterator)

        subtasks_complete = 0
        agent_ptr = 0
        info = {}
        term = False
        action = None
        lower_level_action = None
        while True:
            success = state.ptr is None
            self.success_count += success

            term = term or success or state.term
            if self.lower_level == "train-alone":
                reward = 1 if state.subtask_complete else 0
            else:
                reward = int(success)
            subtasks_complete += state.subtask_complete
            if term:
                if not success and self.break_on_fail:
                    import ipdb

                    ipdb.set_trace()

                info.update(
                    instruction_len=len(lines),
                )
                if not use_failure_buf:
                    info.update(success_without_failure_buf=success)
                if success:
                    info.update(success_line=len(lines), progress=1)
                else:
                    info.update(
                        success_line=state.prev, progress=state.prev / len(lines)
                    )
                subtasks_attempted = subtasks_complete + (not success)
                info.update(
                    subtasks_complete=subtasks_complete,
                    subtasks_attempted=subtasks_attempted,
                )

            info.update(
                subtask_complete=state.subtask_complete,
            )

            def render():
                self.render_instruction(
                    term=term,
                    success=success,
                    lines=lines,
                    state=state,
                    agent_ptr=agent_ptr,
                )
                self.render_world(
                    state=state,
                    action=action,
                    lower_level_action=lower_level_action,
                    reward=reward,
                )

            self._render = render
            obs = state.obs
            pads = [Padding(0)] * (self.n_lines - len(lines))
            padded = lines + pads
            preprocessed_lines = [self.preprocess_line(p) for p in padded]
            mask = [int(isinstance(l, Padding)) for l in padded]
            truthy = [
                self.evaluate_line(l, None, state.counts)
                if agent_ptr < len(lines)
                else 2
                for l in lines
            ]
            truthy = [2 if t is None else int(t) for t in truthy]
            truthy += [3] * (self.n_lines - len(truthy))

            inventory = self.inventory_representation(state)
            obs = Obs(
                obs=obs,
                lines=preprocessed_lines,
                mask=mask,
                active=self.n_lines if state.ptr is None else state.ptr,
                inventory=inventory,
                subtask_complete=state.subtask_complete,
                truthy=truthy,
            )
            self.r.set(f"{self.i},obs", pickle.dumps(obs))
            # if not self.observation_space.contains(obs):
            #     import ipdb
            #
            #     ipdb.set_trace()
            #     self.observation_space.contains(obs)
            obs = OrderedDict(obs._asdict())

            line_specific_info = {
                f"{k}_{10 * (len(lines) // 10)}": v for k, v in info.items()
            }
            action = (yield obs, reward, term, dict(**info, **line_specific_info))
            if action.size == 1:
                action = Action(upper=0, lower=action, delta=0, dg=0, ptr=0)

            action = Action(*action)
            action, lower_level_action, agent_ptr = (
                int(action.upper),
                int(action.lower),
                int(action.ptr),
            )

            info = dict(
                use_failure_buf=use_failure_buf,
                len_failure_buffer=len(self.failure_buffer),
            )

            if action == self.num_subtasks:
                n += 1
                no_op_limit = 200 if self.evaluating else self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(lines)
                if n >= no_op_limit:
                    term = True
            elif state.ptr is not None:
                step += 1
                # noinspection PyUnresolvedReferences
                state = state_iterator.send((action, lower_level_action))

    def inventory_representation(self, state):
        return np.array([state.inventory[i] for i in self.items])

    @property
    def eval_condition_size(self):
        return self._eval_condition_size and self.evaluating

    @staticmethod
    def line_generator(lines):
        line_transitions = defaultdict(list)

        def get_transitions():
            conditions = []
            for i, line in enumerate(lines):
                yield from line.transitions(i, conditions)

        for _from, _to in get_transitions():
            line_transitions[_from].append(_to)
        i = 0
        if_evaluations = []
        while True:
            condition_bit = yield None if i >= len(lines) else i
            if type(lines[i]) is Else:
                evaluation = not if_evaluations.pop()
            else:
                evaluation = bool(condition_bit)
            if type(lines[i]) is If:
                if_evaluations.append(evaluation)
            i = line_transitions[i][evaluation]

    def seed(self, seed=None):
        assert self.seed == seed

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("pause")


def add_arguments(p):
    p.add_argument("--min-lines", type=int)
    p.add_argument("--max-lines", type=int)
    p.add_argument("--max-loops", type=int, default=3)
    p.add_argument("--no-op-limit", type=int)
    p.add_argument("--eval-condition-size", action="store_true")
    p.add_argument("--single-control-flow-type", action="store_true")
    p.add_argument("--max-nesting-depth", type=int, default=1)
    p.add_argument("--subtasks-only", action="store_true")
    p.add_argument("--break-on-fail", action="store_true")
    p.add_argument("--time-to-waste", type=int)
    p.add_argument(
        "--control-flow-types",
        nargs="*",
        type=lambda s: dict(Subtask=Subtask, If=If, Else=Else, While=While, Loop=Loop)[
            s
        ],
    )
    p.add_argument("--no-water", dest="use_water", action="store_false")
    p.add_argument("--1condition", dest="one_condition", action="store_true")
    p.add_argument("--long-jump", action="store_true")
    p.add_argument("--max-failure-sample-prob", type=float)
    p.add_argument("--failure-buffer-size", type=int)
    p.add_argument("--reject-while-prob", type=float)
    p.add_argument("--max-world-resamples", type=int)
    p.add_argument("--max-while-loops", type=int)
    p.add_argument("--world-size", type=int)
    p.add_argument("--term-on", nargs="+", choices=[Env.sell, Env.mine, Env.goto])


def main(env):
    # for i, l in enumerate(env.lower_level_actions):
    # print(i, l)
    actions = [x if type(x) is str else tuple(x) for x in env.lower_level_actions]
    mapping = dict(
        w=(-1, 0), s=(1, 0), a=(0, -1), d=(0, 1), m="mine", l="sell", g="goto"
    )
    mapping2 = {}
    for k, v in mapping.items():
        try:
            mapping2[k] = actions.index(v)
        except ValueError:
            pass

    def action_fn(string):
        action = mapping2.get(string, None)
        if action is None:
            return None
        return np.array(Action(upper=0, lower=action, delta=0, dg=0, ptr=0))

    keyboard_control.run(env, action_fn=action_fn)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--min-eval-lines", type=int)
    PARSER.add_argument("--max-eval-lines", type=int)
    add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    main(Env(rank=0, lower_level="train-alone", **hierarchical_parse_args(PARSER)))
