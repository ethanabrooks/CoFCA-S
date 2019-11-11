from abc import ABC
from collections import defaultdict, namedtuple

import numpy as np
from gym.utils import seeding
from gym.vector.utils import spaces
from rl_utils import hierarchical_parse_args, gym

from ppo import keyboard_control
from ppo.control_flow.lines import If, Else, EndIf, While, EndWhile, Subtask, Padding

Obs = namedtuple("Obs", "action condition lines")
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
    ):
        super().__init__()
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
        self.prev = None
        self.active = None
        self.condition_bit = None
        self.evaluating = False
        self.failing = False
        self.lines = None
        self.line_transitions = None
        self.active_line = None
        self.if_evaluations = None
        self.line_types = [If, Else, EndIf, While, EndWhile, Subtask, Padding]
        self.line_state_transitions = dict(
            initial={If: "following_if", While: "following_while", Subtask: "initial"},
            following_if={Subtask: "inside_if"},
            inside_if={Subtask: "inside_if", Else: "following_else", EndIf: "initial"},
            following_else={Subtask: "inside_else"},
            inside_else={Subtask: "inside_else", EndIf: "initial"},
            following_while={Subtask: "inside_while"},
            inside_while={Subtask: "inside_while", EndWhile: "initial"},
        )
        self.legal_last_lines = dict(
            initial=Subtask, inside_if=EndIf, inside_else=EndIf, inside_while=EndWhile
        )
        if baseline:
            self.action_space = spaces.Discrete(2 * n_lines)
            self.observation_space = spaces.MultiBinary(
                2 + len(self.line_types) * n_lines + (n_lines * 2)
            )
            self.eye = Obs(
                condition=np.eye(2),
                lines=np.eye(len(self.line_types)),
                action=np.eye(n_lines * 2),
            )
        else:
            self.action_space = spaces.MultiDiscrete(np.array([2 * n_lines, n_lines]))
            self.observation_space = spaces.Dict(
                dict(
                    condition=spaces.Discrete(2),
                    lines=spaces.MultiDiscrete(
                        np.array([len(self.line_types)] * n_lines)
                    ),
                    action=spaces.Discrete(n_lines * 2),
                )
            )
        self.t = None

    def reset(self):
        self.last = Last(
            action=(0, 0), selected=0, active=None, reward=None, terminal=None
        )
        self.failing = False
        self.t = 0
        self.condition_bit = self.random.randint(0, 2)
        if self.evaluating:
            assert self.eval_lines is not None
            n_lines = self.eval_lines
        else:
            n_lines = self.random.random_integers(self.min_lines, self.max_lines)
        self.lines = self.get_lines(n_lines, line_state="initial")
        self.line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(self.lines))):
            self.line_transitions[_from].append(_to)
        self.if_evaluations = []
        self.active = 0
        self.prev = 0
        return self.get_observation(action=0)

    def step(self, action):
        s, r, t, i = self._step(action)
        if not self.baseline:
            action = action[0]
        if self.active is None:
            selected = None
        else:
            selected = self.active + action - self.n_lines
        self.last = Last(
            action=action, active=self.active, reward=r, terminal=t, selected=selected
        )
        return s, r, t, i

    def _step(self, action):
        self.t += 1
        if not self.baseline:
            action = int(action[0])
        selected = self.prev + action - self.n_lines
        if selected == len(self.lines):
            # no-op
            return self.get_observation(action), 0, False, {}
        if selected != self.active:
            self.failing = True
            if not self.delayed_reward:
                return self.get_observation(action), 0, True, {}
        self.condition_bit = 1 - int(self.random.rand() < self.flip_prob)
        r = 0
        t = self.t > self.time_limit
        self.prev = self.active
        self.active = self.next()
        if self.active is None:
            r = 1
            if self.delayed_reward and self.failing:
                r = 0
            t = True
        return self.get_observation(action), r, t, {}

    def get_observation(self, action):
        padded = self.lines + [Padding] * (self.n_lines - len(self.lines))
        lines = [self.line_types.index(t) for t in padded]
        obs = Obs(condition=self.condition_bit, lines=lines, action=action)
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
            if line in [Subtask, EndIf]:
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
                yield prev, current + 1  # False: While -> EndWhile
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

    def render(self, mode="human"):
        for i, string in enumerate(self.line_strings(index=0, level=1)):
            print(f"{i}{string}")
        print("Condition:", self.condition_bit)
        print("Reward:", self.last.reward)
        input("pause")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-lines", default=6, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    args = hierarchical_parse_args(parser)
    keyboard_control.run(Env(**args), actions="".join(map(str, range(args["n_lines"]))))
