import abc
from abc import ABC
from collections import defaultdict

from gym.utils import seeding
from ppo import subtasks

from ppo.subtasks.lines import (
    If,
    Else,
    EndIf,
    While,
    EndWhile,
    Subtask,
    initial,
    following_if,
    inside_if,
    following_else,
    inside_else,
    following_while,
    inside_while,
)


class Env(subtasks.Env, ABC):
    def __init__(self, seed, n_subtasks):
        super().__init__()
        self.n_lines = n_subtasks
        self.random, self.seed = seeding.np_random(seed)
        self.lines = None
        self.line_transitions = None
        self.active_line = None
        self.line_state_transitions = {
            initial: {If: following_if, While: following_while, Subtask: initial},
            following_if: {Subtask: inside_if},
            inside_if: {Subtask: inside_if, Else: following_else, EndIf: initial},
            following_else: {Subtask: inside_else, EndIf: initial},
            inside_else: {Subtask: inside_else, EndIf: initial},
            following_while: {Subtask: inside_while},
            inside_while: {Subtask: inside_while, EndWhile: initial},
        }
        self.legal_last_lines = {
            initial: Subtask,
            inside_if: EndIf,
            inside_else: EndIf,
            inside_while: EndWhile,
        }

    def seed(self, seed=None):
        assert self.seed == seed

    def get_lines(self, n, line_state):
        if n == 1:
            try:
                return [self.legal_last_lines[line_state]]
            except IndexError:
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
            current, line = next(lines_iter)
            if line is Subtask:
                yield current - 1, current  # False
                yield current - 1, current  # True
            elif line is If:
                yield from self.get_transitions(lines_iter, current)  # from = If
            elif line is Else:
                assert prev is not None
                yield prev, current + 1  # False: If -> Else + 1
                yield prev, prev + 1  # True: If -> If + 1
                yield from self.get_transitions(lines_iter, current)  # from = Else
            elif line is EndIf:
                assert prev is not None
                yield prev, current + 1  # False: If/Else -> EndIf + 1
                yield prev, prev + 1  # True: If/Else -> If/Else + 1
                return
            elif line is While:
                yield from self.get_transitions(lines_iter, current)  # from = While
            elif line is EndWhile:
                # TODO: should we cycle back to while after EndWhile?
                yield prev, current + 1  # False: While -> EndWhile + 1
                yield prev, prev + 1  # True: While -> While + 1
                yield current, prev + 1  # False: EndWhile -> While + 1
                yield current, current + 1  # True: EndWhile -> EndWhile + 1
                return

    def reset(self):
        self.lines = self.get_lines(self.n_lines, line_state="initial")
        self.line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(self.lines))):
            self.line_transitions[_from].append(_to)
        self.active = self.initial()
        return super().reset()

    def next(self):
        return self.line_transitions[self.active][self.evaluate_condition()]

    def initial(self):
        return 0

    @abc.abstractmethod
    def evaluate_condition(self):
        pass
