from abc import ABC
from collections import defaultdict, namedtuple

import numpy as np
from gym.utils import seeding
from gym.vector.utils import spaces
from rl_utils import hierarchical_parse_args, gym

from ppo import keyboard_control
from ppo.control_flow.lines import If, Else, EndIf, While, EndWhile, Subtask, Padding

Obs = namedtuple("Obs", "active condition lines")
Last = namedtuple("Last", "action active reward terminal selected")


class Env(gym.Env, ABC):
    pairs = {If: EndIf, Else: EndIf, While: EndWhile}

    def __init__(
        self,
        seed,
        min_lines,
        max_lines,
        eval_lines,
        flip_prob,
        time_limit,
        baseline,
        delayed_reward,
        line_types,
        num_subtasks,
    ):
        super().__init__()
        self.num_subtasks = num_subtasks
        self.delayed_reward = delayed_reward
        self.eval_lines = eval_lines
        self.min_lines = min_lines
        self.max_lines = max_lines
        if eval_lines is None:
            self.n_lines = n_lines = self.max_lines
        else:
            assert eval_lines >= self.max_lines
            self.n_lines = n_lines = eval_lines
        self.n_lines += 1
        self.random, self.seed = seeding.np_random(seed)
        self.time_limit = time_limit
        self.flip_prob = flip_prob

        self.baseline = baseline
        self.last = None
        self.active = None
        self.condition_bit = None
        self.evaluating = False
        self.failing = False
        self.lines = None
        self.line_transitions = None
        self.active_line = None
        self.if_evaluations = None
        self.line_types = [If, Else, EndIf, While, EndWhile, Subtask, Padding]
        if baseline:
            self.action_space = spaces.Discrete(2 * self.n_lines)
            self.observation_space = spaces.MultiBinary(
                2 + len(self.line_types) * self.n_lines + (self.n_lines + 1)
            )
            self.eye = Obs(
                condition=np.eye(2),
                lines=np.eye(len(self.line_types)),
                active=np.eye(self.n_lines + 1),
            )
        else:
            self.action_space = spaces.MultiDiscrete(
                np.array([2 * self.n_lines, self.n_lines])
            )
            self.observation_space = spaces.Dict(
                dict(
                    condition=spaces.Discrete(2),
                    lines=spaces.MultiDiscrete(
                        np.array([len(self.line_types)] * self.n_lines)
                    ),
                    active=spaces.Discrete(self.n_lines + 1),
                )
            )
        self.t = None

    def reset(self):
        self.last = Last(
            action=None, selected=0, active=None, reward=None, terminal=None
        )
        self.failing = False
        self.t = 0
        self.condition_bit = self.random.randint(0, 2)
        if self.evaluating:
            assert self.eval_lines is not None
            n_lines = self.eval_lines
        else:
            n_lines = self.random.random_integers(self.min_lines, self.max_lines)
        self.lines = self.get_lines(n_lines, active_conditions=[])

        self.line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(self.lines)), []):
            self.line_transitions[_from].append(_to)
        self.if_evaluations = []
        self.active = 0
        return self.get_observation(action=0)

    def step(self, action):
        if not self.baseline:
            action = int(action[0])
        delta = action - self.n_lines
        selected = self.active + delta
        s, r, t, i = self._step(action=action, selected=selected)
        self.last = Last(
            action=action, active=self.active, reward=r, terminal=t, selected=selected
        )
        return s, r, t, i

    def _step(self, action, selected):
        i = {}
        if self.t == 0:
            i.update(
                if_lines=self.lines.count(If),
                else_lines=self.lines.count(Else),
                while_lines=self.lines.count(While),
                nesting_depth=self.get_nesting_depth(),
            )
            keys = {
                (If, EndIf): "if clause length",
                (If, Else): "if-else clause length",
                (Else, EndIf): "else clause length",
                (While, EndWhile): "while clause length",
            }
            for k, v in self.average_interval():
                i[keys[k]] = v
        prev = self.lines[self.active]
        self.t += 1
        self.active = self.next()
        self.condition_bit = 1 - int(self.random.rand() < self.flip_prob)
        r = 0
        t = self.t > self.time_limit
        if self.active is None:
            if self.delayed_reward and self.failing:
                r = 0
            else:
                r = 1
            t = True
        elif selected != self.active:
            i.update(termination_line=self.active)
            self.failing = True
            if not self.delayed_reward:
                r = 0
                t = True
        if prev is While:
            i.update(successful_while=not self.failing)
        if prev is If:
            i.update(successful_if=not self.failing)
        return self.get_observation(action), r, t, i

    def average_interval(self):
        intervals = defaultdict(lambda: [None])
        pairs = [(If, EndIf), (While, EndWhile)]
        if Else in self.lines:
            pairs.extend([(If, Else), (Else, EndIf)])
        for line in self.lines:
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

    def get_observation(self, action):
        padded = self.lines + [Padding] * (self.n_lines - len(self.lines))
        lines = [self.line_types.index(t) for t in padded]
        obs = Obs(
            condition=self.condition_bit,
            lines=lines,
            active=self.n_lines if self.active is None else self.active,
        )
        if self.baseline:
            obs = [eye[o].flatten() for eye, o in zip(self.eye, obs)]
            obs = np.concatenate(obs)
        else:
            obs = obs._asdict()
        # assert self.observation_space.contains(obs)
        return obs

    def seed(self, seed=None):
        assert self.seed == seed

    def get_nesting_depth(self):
        max_depth = 0
        depth = 0
        for line in self.lines:
            if line in [If, While]:
                depth += 1
            if line in [EndIf, EndWhile]:
                depth -= 1
            max_depth = max(depth, max_depth)
        return max_depth

    def get_lines(self, n, active_conditions, last=None):
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
        if n > len(active_conditions) + 2:
            line_types += [If, While]
        if active_conditions and last is Subtask:
            last_condition = active_conditions[-1]
            if last_condition is If:
                line_types += [Else, EndIf]
            elif last_condition is Else:
                line_types += [EndIf]
            elif last_condition is While:
                line_types += [EndWhile]
        line_type = self.random.choice(line_types)
        if line_type in [If, While]:
            active_conditions = active_conditions + [line_type]
        elif line_type is Else:
            active_conditions = active_conditions[:-1] + [line_type]
        elif line_type in [EndIf, EndWhile]:
            active_conditions = active_conditions[:-1]
        get_lines = self.get_lines(
            n - 1, active_conditions=active_conditions, last=line_type
        )
        return [line_type] + get_lines

    def get_transitions(self, lines_iter, previous):
        while True:  # stops at StopIteration
            try:
                current, line = next(lines_iter)
            except StopIteration:
                return
            if line in [Subtask, EndIf]:
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

    def next(self, i=None):
        if i is None:
            i = self.active
        evaluation = self.evaluate_condition(i)
        if self.lines[i] is If:
            self.if_evaluations.append(evaluation)
        i = self.line_transitions[i][evaluation]
        if i >= len(self.lines):
            return None
        return i

    def evaluate_condition(self, i=None):
        if i is None:
            i = self.active
        if self.lines[i] is Else:
            return not self.if_evaluations.pop()
        return bool(self.condition_bit)

    def train(self):
        self.evaluating = False

    def evaluate(self):
        self.evaluating = True

    def line_strings(self, index, level):
        if index == len(self.lines):
            return
        line = self.lines[index]
        if line in [Else, EndIf, EndWhile]:
            level -= 1
        if index == self.active and index == self.last.selected:
            pre = "+ "
        elif index == self.last.selected:
            pre = "- "
        elif index == self.active:
            pre = "| "
        else:
            pre = "  "
        indent = pre * level
        yield indent + line.__name__
        if line in [If, While, Else]:
            level += 1
        yield from self.line_strings(index + 1, level)

    def render(self, mode="human", pause=True):
        for i, string in enumerate(self.line_strings(index=0, level=1)):
            print(f"{i}{string}")
        print("Condition:", self.condition_bit)
        print(self.last)
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
    parser.add_argument("--num-subtasks", default=12, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    parser.add_argument("--delayed-reward", action="store_true")
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(
        Env(**args, baseline=False, line_types=None), action_fn=action_fn
    )
