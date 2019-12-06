import itertools
from abc import ABC
from collections import defaultdict, namedtuple, OrderedDict

import numpy as np
from gym.utils import seeding
from gym.vector.utils import spaces
from rl_utils import hierarchical_parse_args, gym

from ppo import keyboard_control
from ppo.control_flow.lines import If, Else, EndIf, While, EndWhile, Subtask, Padding

Obs = namedtuple("Obs", "condition lines")


class Env(gym.Env, ABC):
    pairs = {If: EndIf, Else: EndIf, While: EndWhile}
    line_types = [If, Else, EndIf, While, EndWhile, Subtask, Padding]

    def __init__(
        self,
        seed,
        min_lines,
        max_lines,
        eval_lines,
        flip_prob,
        time_limit,
        terminate_on_failure,
        num_subtasks,
        max_nesting_depth,
        eval_condition_size,
        no_op_limit,
        evaluating,
        baseline=False,
    ):
        super().__init__()
        self.terminate_on_failure = terminate_on_failure
        self.no_op_limit = no_op_limit
        self.eval_condition_size = eval_condition_size
        self.max_nesting_depth = max_nesting_depth
        self.num_subtasks = num_subtasks
        self.eval_lines = eval_lines
        self.min_lines = min_lines
        self.max_lines = max_lines
        if evaluating:
            self.n_lines = eval_lines
        else:
            self.n_lines = max_lines
        # self.n_lines += 1
        self.random, self.seed = seeding.np_random(seed)
        self.time_limit = time_limit
        self.flip_prob = flip_prob
        self.baseline = baseline
        self.evaluating = False
        self.last = None
        self.iterator = None
        self._render = None
        if baseline:
            raise NotImplementedError
            self.action_space = spaces.Discrete(self.num_subtasks + 1)
            n_line_types = len(self.line_types) + num_subtasks
            self.observation_space = spaces.Dict(
                dict(
                    condition=spaces.Discrete(2),
                    lines=spaces.MultiBinary(n_line_types * self.n_lines),
                )
            )
            self.eye = np.eye(n_line_types)
        else:
            self.action_space = spaces.MultiDiscrete(
                np.array([self.num_subtasks + 1, 2 * self.n_lines])
            )
            self.observation_space = spaces.Dict(
                dict(
                    condition=spaces.Discrete(2),
                    lines=spaces.MultiDiscrete(
                        np.array([len(self.line_types) + num_subtasks] * self.n_lines)
                    ),
                )
            )

    def reset(self):
        self.iterator = self.generator(evaluating=self.evaluating)
        s, r, t, i = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def generator(self, evaluating):
        n = 0
        eval_condition_size = self.eval_condition_size and self.evaluating
        condition_bit = 0 if eval_condition_size else self.random.randint(0, 2)
        lines = list(self.build_lines(eval_condition_size, evaluating))
        line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(lines)), []):
            line_transitions[_from].append(_to)
        line_iterator = self.line_generator(lines, line_transitions)

        def next_subtask(b):
            while True:
                try:
                    # allow b=None for first send in order to initialize line_iterator
                    a = line_iterator.send(b)
                    if type(lines[a]) is Subtask:
                        return a
                    if b is None:
                        b = condition_bit
                except StopIteration:
                    return None

        prev, active = 0, next_subtask(None)

        if active is None:
            # skip trivial tasks
            yield from self.generator(evaluating)
            return

        i = self.get_task_info(lines)
        failing = False
        selected = 0
        for _ in itertools.count() if evaluating else range(self.time_limit):
            r = int(active is None and not failing)
            t = active is None or (self.terminate_on_failure and failing)

            def strings(index, level):
                if index == len(lines):
                    return
                line = lines[index]
                if line in [Else, EndIf, EndWhile]:
                    level -= 1
                if index == active and index == selected:
                    pre = "+ "
                elif index == selected:
                    pre = "- "
                elif index == active:
                    pre = "| "
                else:
                    pre = "  "
                indent = pre * level
                if type(line) is Subtask:
                    yield f"{indent}Subtask {line.id}"
                else:
                    yield f"{indent}{line.__name__}"
                if line in [If, While, Else]:
                    level += 1
                yield from strings(index=index + 1, level=level)

            def render():
                for i, string in enumerate(strings(index=0, level=1)):
                    print(f"{i}{string}")
                print("Condition:", condition_bit)
                print("Failing:", failing)
                print("Reward:", r)
                # print("Terminal:", t)
                # print("Time step:", step)
                # print("No ops:", n)

            self._render = render

            action = yield self.get_observation(lines, condition_bit), r, t, i
            if self.baseline:
                selected = None
            else:
                action, delta = action
                selected = (selected + delta - self.n_lines) % self.n_lines

            if action == self.num_subtasks:
                n += 1
                r = 0
                t = self.no_op_limit and n == self.no_op_limit
            else:
                success = active is None
                if not (failing or success):
                    failing = action != lines[active].id

                i = {}
                if success:
                    i.update(success_line=len(lines))
                elif failing:
                    i.update(success_line=prev, failure_line=active)

                if self.random.rand() < self.flip_prob:
                    condition_bit = int(not condition_bit)

                prev, active = active, next_subtask(condition_bit)

        # out of time
        yield self.get_observation(lines, condition_bit), 0, True, i

    def build_lines(self, eval_condition_size, evaluating):
        n_lines = (
            self.eval_lines
            if evaluating
            else self.random.randint(self.min_lines, self.max_lines + 1)
        )
        assert n_lines is not None
        if eval_condition_size:
            line0 = self.random.choice([While, If])
            edge_length = self.random.randint(self.max_lines, self.eval_lines)
            lines = [line0] + [Subtask] * (edge_length - 2)
            lines += [EndWhile if line0 is While else EndIf, Subtask]
        else:
            lines = self.get_lines(
                n_lines, active_conditions=[], max_nesting_depth=self.max_nesting_depth
            )
        for line in lines:
            if line is Subtask:
                yield Subtask(self.random.choice(self.num_subtasks))
            else:
                yield line

    def get_lines(
        self, n, active_conditions, last=None, nesting_depth=0, max_nesting_depth=None
    ):
        if n < 0:
            return []
        if n == 0:
            return []
        if n == len(active_conditions):
            lines = [self.pairs[c] for c in reversed(active_conditions)]
            # noinspection PyTypeChecker
            return lines + [Subtask for _ in range(n - len(lines))]
        elif n == 1:
            return [Subtask]
        line_types = [Subtask]
        enough_space = (
            n > len(active_conditions) + 2
        )  # + 2 if for the If/While and the subsequent Subtask
        if enough_space and (
            max_nesting_depth is None or nesting_depth < max_nesting_depth
        ):
            line_types += [If, While]
        if active_conditions and last is Subtask:
            last_condition = active_conditions[-1]
            if last_condition is If and enough_space:
                line_types += [Else]
            elif last_condition in [If, Else]:
                line_types += [EndIf]
            elif last_condition is While:
                line_types += [EndWhile]
        line_type = self.random.choice(line_types)
        if line_type in [If, While]:
            active_conditions = active_conditions + [line_type]
            nesting_depth += 1
        elif line_type is Else:
            active_conditions = active_conditions[:-1] + [line_type]
        elif line_type in [EndIf, EndWhile]:
            active_conditions = active_conditions[:-1]
            nesting_depth -= 1
        get_lines = self.get_lines(
            n - 1,
            active_conditions=active_conditions,
            last=line_type,
            nesting_depth=nesting_depth,
            max_nesting_depth=max_nesting_depth,
        )
        return [line_type] + get_lines

    def get_transitions(self, lines_iter, previous):
        while True:  # stops at StopIteration
            try:
                current, line = next(lines_iter)
            except StopIteration:
                return
            if line is EndIf or type(line) is Subtask:
                yield current, current + 1  # False
                yield current, current + 1  # True
            if line is If:
                yield from self.get_transitions(
                    lines_iter, previous + [current]
                )  # from = If
            elif line is Else:
                prev = previous[-1]
                yield prev, current  # False: If -> Else
                yield prev, prev + 1  # True: If -> If + 1
                previous[-1] = current
            elif line is EndIf:
                prev = previous[-1]
                yield prev, current  # False: If/Else -> EndIf
                yield prev, prev + 1  # True: If/Else -> If/Else + 1
                return
            elif line is While:
                yield from self.get_transitions(
                    lines_iter, previous + [current]
                )  # from = While
            elif line is EndWhile:
                prev = previous[-1]
                # While
                yield prev, current + 1  # False: While -> EndWhile + 1
                yield prev, prev + 1  # True: While -> While + 1
                # EndWhile
                yield current, prev  # False: EndWhile -> While
                yield current, prev  # True: EndWhile -> While
                return

    def get_task_info(self, lines):
        num_if = lines.count(If)
        num_else = lines.count(Else)
        num_while = lines.count(While)
        num_subtask = lines.count(lambda l: type(l) is Subtask)
        i = dict(
            if_lines=num_if,
            else_lines=num_else,
            while_lines=num_while,
            nesting_depth=self.get_nesting_depth(lines),
            num_edges=2 * (num_if + num_else + num_while) + num_subtask,
        )
        keys = {
            (If, EndIf): "if clause length",
            (If, Else): "if-else clause length",
            (Else, EndIf): "else clause length",
            (While, EndWhile): "while clause length",
        }
        for k, v in self.average_interval(lines):
            i[keys[k]] = v
        return i

    def get_observation(self, lines, condition_bit):
        padded = lines + [Padding] * (self.n_lines - len(lines))
        lines = [
            t.id if type(t) is Subtask else self.num_subtasks + self.line_types.index(t)
            for t in padded
        ]
        obs = Obs(condition=condition_bit, lines=lines)
        if self.baseline:
            obs = Obs(condition=obs.condition, lines=self.eye[obs.lines].flatten())
        else:
            obs = obs._asdict()
        assert self.observation_space.contains(obs)
        return obs

    @staticmethod
    def average_interval(lines):
        intervals = defaultdict(lambda: [None])
        pairs = [(If, EndIf), (While, EndWhile)]
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
            if line in [If, While]:
                depth += 1
            if line in [EndIf, EndWhile]:
                depth -= 1
            max_depth = max(depth, max_depth)
        return max_depth

    def seed(self, seed=None):
        assert self.seed == seed

    @staticmethod
    def line_generator(lines, line_transitions, i=0):
        if_evaluations = []
        while i < len(lines):
            bit = yield i
            if lines[i] is If:
                if_evaluations.append(bit)
            b = (not if_evaluations.pop()) if lines[i] is Else else bit
            i = line_transitions[i][b]

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("pause")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--min-lines", default=6, type=int)
    parser.add_argument("--max-lines", default=6, type=int)
    parser.add_argument("--eval-lines", type=int)
    parser.add_argument("--time-limit", default=100, type=int)
    parser.add_argument("--no-op-limit", default=100, type=int)
    parser.add_argument("--num-subtasks", default=12, type=int)
    parser.add_argument("--max-nesting-depth", default=2, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    parser.add_argument("--terminate-on-failure", action="store_true")
    parser.add_argument("--eval-condition-size", action="store_true")
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(Env(**args, baseline=False), action_fn=action_fn)
