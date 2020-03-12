import functools
from abc import ABC
from collections import defaultdict, namedtuple, OrderedDict, Counter

import numpy as np

# import skimage.draw
from gym.utils import seeding
from gym.vector.utils import spaces
from rl_utils import hierarchical_parse_args, gym


from ppo import keyboard_control
from ppo.control_flow.lines import (
    If,
    Else,
    EndIf,
    While,
    EndWhile,
    Subtask,
    Padding,
    Line,
    Loop,
    EndLoop,
)
from ppo.utils import RED, RESET, GREEN

Obs = namedtuple("Obs", "active lines obs")
Last = namedtuple("Last", "action active reward terminal selected")
State = namedtuple("State", "obs condition prev ptr condition_evaluations term")


class Env(gym.Env, ABC):
    pairs = {If: EndIf, Else: EndIf, While: EndWhile, Loop: EndLoop}
    line_types = [If, Else, EndIf, While, EndWhile, EndLoop, Subtask, Padding, Loop]

    def __init__(
        self,
        min_lines,
        max_lines,
        flip_prob,
        num_subtasks,
        max_nesting_depth,
        eval_condition_size,
        single_control_flow_type,
        no_op_limit,
        time_to_waste,
        subtasks_only,
        break_on_fail,
        max_loops,
        rank,
        control_flow_types,
        seed=0,
        eval_lines=None,
        evaluating=False,
    ):
        super().__init__()
        self.control_flow_types = control_flow_types
        self.rank = rank
        self.max_loops = max_loops
        self.break_on_fail = break_on_fail
        self.subtasks_only = subtasks_only
        self.no_op_limit = no_op_limit
        self._eval_condition_size = eval_condition_size
        self._single_control_flow_type = single_control_flow_type
        self.max_nesting_depth = max_nesting_depth
        self.num_subtasks = num_subtasks
        self.time_to_waste = time_to_waste
        self.time_remaining = None

        self.loops = None
        self.eval_lines = eval_lines
        self.min_lines = min_lines
        self.max_lines = max_lines
        if evaluating:
            self.n_lines = eval_lines
        else:
            self.n_lines = max_lines
        self.n_lines += 1
        self.random, self.seed = seeding.np_random(seed)
        self.flip_prob = flip_prob
        self.evaluating = evaluating
        self.iterator = None
        self._render = None
        self.action_space = spaces.MultiDiscrete(
            np.array([self.num_subtasks + 1, 2 * self.n_lines, self.n_lines])
        )

        def possible_lines():
            for i in range(num_subtasks):
                yield Subtask(i)
            for i in range(1, max_loops + 1):
                yield Loop(i)
            for line_type in self.line_types:
                if line_type not in (Subtask, Loop):
                    yield line_type(0)

        self.possible_lines = list(possible_lines())
        self.observation_space = spaces.Dict(
            dict(
                obs=spaces.Discrete(2),
                lines=spaces.MultiDiscrete(
                    np.array([len(self.possible_lines)] * self.n_lines)
                ),
                active=spaces.Discrete(self.n_lines + 1),
            )
        )

    def reset(self):
        self.iterator = self.generator()
        s, r, t, i = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def generator(self):
        step = 0
        n = 0
        lines = self.build_lines()
        state_iterator = self.state_generator(lines)
        state = next(state_iterator)
        actions = []
        program_counter = []
        evaluations = []

        agent_ptr = 0
        info = {}
        term = False
        action = None
        while True:
            if state.ptr is not None:
                program_counter.append(state.ptr)
            success = state.ptr is None
            reward = int(success)
            if success:
                info.update(success_line=len(lines))

            term = term or success or state.term
            if term:
                if not success and self.break_on_fail:
                    import ipdb

                    ipdb.set_trace()

                info.update(
                    instruction=[self.preprocess_line(l) for l in lines],
                    actions=actions,
                    program_counter=program_counter,
                    evaluations=evaluations,
                )

            info.update(regret=1 if term and not success else 0)

            def line_strings(index, level):
                if index == len(lines):
                    return
                line = lines[index]
                if index == state.ptr and index == agent_ptr:
                    pre = "+ "
                elif index == agent_ptr:
                    pre = "- "
                elif index == state.ptr:
                    pre = "| "
                else:
                    pre = "  "
                if line.depth_change < 0:
                    level += line.depth_change
                indent = pre * level
                if line.depth_change > 0:
                    level += line.depth_change
                # if type(line) is Subtask:
                yield f"{indent}{line}"
                # else:
                #     yield f"{indent}{line.__name__}"
                # if line in [If, While, Else]:
                yield from line_strings(index + 1, level)

            def render():
                if term:
                    print(GREEN if success else RED)
                for i, string in enumerate(line_strings(index=0, level=1)):
                    print(f"{i}{string}")
                print("Selected:", agent_ptr)
                print("Action:", action)
                print("Reward", reward)
                print("Obs:")
                print(RESET)
                self.print_obs(state.obs)

            self._render = render
            obs = self.get_observation(state.obs, state.ptr, lines)

            action = (yield obs, reward, term, info)
            actions += [list(action.astype(int))]
            action, agent_ptr = int(action[0]), int(action[-1])
            info = {}

            if action == self.num_subtasks:
                n += 1
                no_op_limit = 200 if self.evaluating else self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(lines)
                if n >= no_op_limit:
                    term = True
            elif state.ptr is not None:
                step += 1
                if action != lines[state.ptr].id:
                    info.update(success_line=state.prev, failure_line=state.ptr)
                state = state_iterator.send(action)
                evaluations.extend(state.condition_evaluations)

    @property
    def eval_condition_size(self):
        return self._eval_condition_size and self.evaluating

    @property
    def single_control_flow_type(self):
        return self._single_control_flow_type and not self.evaluating

    def build_lines(self):
        if self.evaluating:
            assert self.eval_lines is not None
            n_lines = self.eval_lines
        else:
            n_lines = self.random.random_integers(self.min_lines, self.max_lines)
        if self.eval_condition_size:
            line0 = self.random.choice([While, If])
            edge_length = self.random.random_integers(
                self.max_lines, self.eval_lines - 1
            )
            lines = [line0] + [Subtask] * (edge_length - 2)
            lines += [EndWhile if line0 is While else EndIf, Subtask]
        else:
            control_flow_types = self.control_flow_types
            if self.single_control_flow_type:
                control_flow_types = [np.random.choice(self.control_flow_types)]
            lines = list(
                Line.generate_lines(
                    n_lines,
                    remaining_depth=self.max_nesting_depth,
                    random=self.random,
                    legal_lines=control_flow_types + [Subtask],
                )
            )
            for l in lines:
                print(l)
        import ipdb

        ipdb.set_trace()
        return list(self.assign_line_ids(lines))

    def assign_line_ids(self, lines):
        for line in lines:
            if line is Subtask:
                yield Subtask(self.random.choice(self.num_subtasks))
            elif line is Loop:
                yield Loop(self.random.randint(1, 1 + self.max_loops))
            else:
                yield line(0)

    @functools.lru_cache(maxsize=120)
    def preprocess_line(self, line):
        return self.possible_lines.index(line)

    def choose_line_types(
        self,
        n,
        active_conditions,
        control_flow_types,
        last=None,
        nesting_depth=0,
        max_nesting_depth=None,
    ):
        if n < 0:
            return []
        if n == 0:
            return []
        if n == len(active_conditions):
            lines = [self.pairs[c] for c in reversed(active_conditions)]
            return lines + [Subtask for _ in range(n - len(lines))]
        elif n == 1:
            return [Subtask]
        line_types = [Subtask]
        enough_space = n > len(active_conditions) + 2
        if (
            enough_space
            and (max_nesting_depth is None or nesting_depth < max_nesting_depth)
            and not self.subtasks_only
        ):
            line_types += control_flow_types
        if active_conditions and last is Subtask:
            last_condition = active_conditions[-1]
            if last_condition is If:
                line_types += [EndIf]
            if last_condition is If and enough_space:
                line_types += [Else]
            elif last_condition is Else:
                line_types += [EndIf]
            elif last_condition is While:
                line_types += [EndWhile]
            elif last_condition is Loop:
                line_types += [EndLoop]
        line_type = self.random.choice(line_types)
        if line_type in [If, While, Loop]:
            active_conditions = active_conditions + [line_type]
            nesting_depth += 1
        elif line_type is Else:
            active_conditions = active_conditions[:-1] + [line_type]
        elif line_type in [EndIf, EndWhile, EndLoop]:
            active_conditions = active_conditions[:-1]
            nesting_depth -= 1
        get_lines = self.choose_line_types(
            n - 1,
            active_conditions=active_conditions,
            last=line_type,
            control_flow_types=control_flow_types,
            nesting_depth=nesting_depth,
            max_nesting_depth=max_nesting_depth,
        )
        return [line_type] + get_lines

    def line_generator(self, lines):
        line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(lines)), []):
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

    def get_transitions(self, lines_iter, previous):
        while True:  # stops at StopIteration
            try:
                current, line = next(lines_iter)
            except StopIteration:
                return
            if type(line) is EndIf or type(line) is Subtask:
                yield current, current + 1  # False
                yield current, current + 1  # True
            if type(line) is If:
                yield from self.get_transitions(
                    lines_iter, previous + [current]
                )  # from = If
            elif type(line) is Else:
                prev = previous[-1]
                yield prev, current  # False: If -> Else
                yield prev, prev + 1  # True: If -> If + 1
                previous[-1] = current
            elif type(line) is EndIf:
                prev = previous[-1]
                yield prev, current  # False: If/Else -> EndIf
                yield prev, prev + 1  # True: If/Else -> If/Else + 1
                return
            elif type(line) in (While, Loop):
                yield from self.get_transitions(
                    lines_iter, previous + [current]
                )  # from = While
            elif type(line) in (EndWhile, EndLoop):
                prev = previous[-1]
                # While
                yield prev, current + 1  # False: While -> EndWhile + 1
                yield prev, prev + 1  # True: While -> While + 1
                # EndWhile
                yield current, prev  # False: EndWhile -> While
                yield current, prev  # True: EndWhile -> While
                return

    def state_generator(self, lines) -> State:
        line_iterator = self.line_generator(lines)
        condition_bit = 0 if self.eval_condition_size else self.random.choice(2)
        condition_evaluations = []
        self.time_remaining = self.time_to_waste
        self.loops = None

        def next_subtask(msg=condition_bit):
            l = line_iterator.send(msg)
            while not (l is None or type(lines[l]) is Subtask):
                line = lines[l]
                if type(line) in (If, While):
                    condition_evaluations.append(condition_bit)
                if type(line) is Loop:
                    if self.loops is None:
                        self.loops = line.id
                    else:
                        self.loops -= 1
                    l = line_iterator.send(self.loops > 0)
                    if self.loops == 0:
                        self.loops = None
                else:
                    l = line_iterator.send(condition_bit)
            self.time_remaining += 1
            return l

        prev, ptr = 0, next_subtask(None)
        term = False
        while True:
            action = yield State(
                obs=condition_bit,
                condition=condition_bit,
                prev=prev,
                ptr=ptr,
                condition_evaluations=condition_evaluations,
                term=term,
            )
            if not self.time_remaining or action != lines[ptr].id:
                term = True
            else:
                self.time_remaining -= 1
                condition_bit = abs(
                    condition_bit - int(self.random.rand() < self.flip_prob)
                )
                prev, ptr = ptr, next_subtask()

    def get_observation(self, obs, active, lines):
        padded = lines + [Padding(0)] * (self.n_lines - len(lines))
        lines = [self.preprocess_line(p) for p in padded]
        return Obs(
            obs=obs, lines=lines, active=self.n_lines if active is None else active
        )._asdict()

    @staticmethod
    def print_obs(obs):
        print(obs)

    @staticmethod
    def average_interval(lines):
        intervals = defaultdict(lambda: [None])
        pairs = [(If, EndIf), (While, EndWhile), (Loop, EndLoop)]
        if Else in lines:
            pairs.extend([(If, Else), (Else, EndIf)])
        for line in lines:
            for start, stop in pairs:
                if line is start:
                    intervals[start, stop][-1] = 0
                if line is stop:
                    intervals[start, stop].append(None)
            for k, (*_, value) in intervals.items():
                if value is not None:
                    intervals[k][-1] += 1
        for keys, values in intervals.items():
            values = [v for v in values if v]
            if values:
                yield keys, sum(values) / len(values)

    @staticmethod
    def get_nesting_depth(lines):
        max_depth = 0
        depth = 0
        for line in lines:
            max_depth = max(line.depth_change, max_depth)
        return max_depth

    def seed(self, seed=None):
        assert self.seed == seed

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("pause")


def build_parser(p):
    p.add_argument("--min-lines", type=int, required=True)
    p.add_argument("--max-lines", type=int, required=True)
    p.add_argument("--num-subtasks", type=int, default=12)
    p.add_argument("--max-loops", type=int, default=2)
    p.add_argument("--no-op-limit", type=int)
    p.add_argument("--flip-prob", type=float, default=0.5)
    p.add_argument("--eval-condition-size", action="store_true")
    p.add_argument("--single-control-flow-type", action="store_true")
    p.add_argument("--max-nesting-depth", type=int)
    p.add_argument("--subtasks-only", action="store_true")
    p.add_argument("--break-on-fail", action="store_true")
    p.add_argument("--time-to-waste", type=int, required=True)
    p.add_argument(
        "--control-flow-types",
        nargs="*",
        type=lambda s: dict(If=If, While=While, Else=Else, Loop=Loop,).get(s),
    )
    return p


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = hierarchical_parse_args(build_parser(parser))

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
