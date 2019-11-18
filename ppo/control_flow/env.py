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
        self.choices = None
        self.target = None
        self.line_types = [If, Else, EndIf, While, EndWhile, Subtask, Padding]
        if line_types is None:
            line_types = "if-while-else"
        initial = {Subtask: "initial"}
        if "if" in line_types:
            initial[If] = "following_if"
        if "while" in line_types:
            initial[While] = "following_while"
        inside_if = {Subtask: "inside_if", EndIf: "initial"}
        if "else" in line_types:
            inside_if[Else] = "following_else"
        self.line_state_transitions = dict(
            initial=initial,
            following_if={Subtask: "inside_if"},
            inside_if=inside_if,
            following_else={Subtask: "inside_else"},
            inside_else={Subtask: "inside_else", EndIf: "initial"},
            following_while={Subtask: "inside_while"},
            inside_while={Subtask: "inside_while", EndWhile: "initial"},
        )
        self.legal_last_lines = dict(
            initial=Subtask, inside_if=EndIf, inside_else=EndIf, inside_while=EndWhile
        )
        if baseline:
            raise NotImplementedError
            self.action_space = spaces.Discrete(2 * n_lines)
            self.observation_space = spaces.MultiBinary(
                2 + len(self.line_types) * n_lines + (n_lines + 1)
            )
            self.eye = Obs(
                condition=np.eye(2),
                lines=np.eye(len(self.line_types)),
                active=np.eye(n_lines + 1),
            )
        else:
            self.action_space = spaces.MultiDiscrete(
                np.array([self.max_lines + 1, 2 * n_lines])
            )
            self.observation_space = spaces.Dict(
                dict(
                    condition=spaces.Discrete(2),
                    lines=spaces.MultiDiscrete(
                        np.array([len(self.line_types) + num_subtasks] * n_lines)
                    ),
                    active=spaces.Discrete(n_lines + 1),
                )
            )
        self.t = None

    def reset(self):
        self.last = Last(
            action=None, selected=0, active=None, reward=None, terminal=None
        )
        self.failing = False
        self.choices = []
        self.target = []
        self.t = 0
        self.condition_bit = self.random.randint(0, 2)
        if self.evaluating:
            assert self.eval_lines is not None
            n_lines = self.eval_lines
        else:
            n_lines = self.random.random_integers(self.min_lines, self.max_lines)
        lines = self.get_lines(n_lines, line_state="initial")
        self.lines = [
            Subtask(self.random.choice(self.num_subtasks)) if line is Subtask else line
            for line in lines
        ]
        self.line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(self.lines))):
            self.line_transitions[_from].append(_to)
        self.if_evaluations = []
        self.active = 0
        return self.get_observation(action=0)

    def step(self, action):
        action, selected = action
        s, r, t, i = self._step(action=int(action))
        self.last = Last(
            action=action, selected=selected, active=self.active, reward=r, terminal=t
        )
        return s, r, t, i

    def _step(self, action):
        i = {}
        if self.t == 0:
            i.update(
                if_lines=self.lines.count(If),
                else_lines=self.lines.count(Else),
                while_lines=self.lines.count(While),
            )
        line = self.lines[self.active]
        if action < self.num_subtasks:
            self.choices.append(action)
        if type(line) is Subtask:
            self.target.append(line.id)
        self.t += 1
        self.active = self.next()
        self.condition_bit = 1 - int(self.random.rand() < self.flip_prob)
        r = 0
        t = self.t > self.time_limit
        if self.active is None:
            r = int(tuple(self.choices) == tuple(self.target))
            t = True
        elif not (
            contains(self.choices, self.target) or contains(self.target, self.choices)
        ):
            i.update(termination_line=self.active)
            self.failing = True
            if not self.delayed_reward:
                r = 0
                t = True
        if line is While:
            i.update(successful_while=not self.failing)
        if line is If:
            i.update(successful_if=not self.failing)
        return self.get_observation(action), r, t, i

    def get_observation(self, action):
        padded = self.lines + [Padding] * (self.n_lines - len(self.lines))
        lines = [
            t.id if type(t) is Subtask else self.num_subtasks + self.line_types.index(t)
            for t in padded
        ]
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
        assert self.observation_space.contains(obs)
        return obs

    def seed(self, seed=None):
        assert self.seed == seed

    def get_lines(self, n, line_state):
        if n == 1:
            try:
                return [self.legal_last_lines[line_state]]
            except KeyError:
                return None

        possible_lines = list(self.line_state_transitions[line_state])
        self.random.shuffle(possible_lines)
        for line in possible_lines:
            new_state = self.line_state_transitions[line_state][line]
            lines = self.get_lines(n - 1, new_state)
            if lines is not None:  # valid last line
                return [line, *lines]

    def get_transitions(self, lines_iter, prev=None):
        while True:  # stops at StopIteration
            try:
                current, line = next(lines_iter)
            except StopIteration:
                return
            if line is EndIf or type(line) is Subtask:
                yield current, current + 1  # False
                yield current, current + 1  # True
            if line is If:
                yield from self.get_transitions(lines_iter, current)  # from = If
            elif line is Else:
                assert prev is not None
                yield prev, current  # False: If -> Else
                yield prev, prev + 1  # True: If -> If + 1
                yield from self.get_transitions(lines_iter, current)  # from = Else
            elif line is EndIf:
                assert prev is not None
                yield prev, current  # False: If/Else -> EndIf
                yield prev, prev + 1  # True: If/Else -> If/Else + 1
                return
            elif line is While:
                yield from self.get_transitions(lines_iter, current)  # from = While
            elif line is EndWhile:
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
        if type(line) is Subtask:
            yield f"{indent}Subtask {line.id}"
        else:
            yield f"{indent}{line.__name__}"
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


def contains(A, B):
    for b in B:
        if b not in A:
            return False
    return True


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
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(
        Env(**args, baseline=False, delayed_reward=False, line_types=None),
        action_fn=action_fn,
    )
