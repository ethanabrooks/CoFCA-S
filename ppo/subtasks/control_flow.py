import abc
from abc import ABC
from collections import defaultdict

from gym.utils import seeding
from ppo import subtasks
from ppo.subtasks.lines import If, Else, EndIf, While, EndWhile, Subtask


class Env(subtasks.Env, ABC):
    def __init__(self, seed, n_subtasks):
        super().__init__()
        self.n_lines = n_subtasks
        self.random, self.seed = seeding.np_random(seed)
        self.lines = None
        self.line_transitions = None
        self.active_line = None
        self.if_evaluations = None
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
                yield prev, current  # False: While -> EndWhile
                yield prev, prev + 1  # True: While -> While + 1
                # EndWhile
                yield current, current + 1  # False: EndWhile -> EndWhile + 1
                yield current, prev  # True: EndWhile -> While1
                return

    def reset(self):
        self.lines = self.get_lines(self.n_lines, line_state="initial")
        self.line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(self.lines))):
            self.line_transitions[_from].append(_to)
        self.if_evaluations = []
        self.active = self.initial()
        if self.active is None:
            return self.reset()
        return super().reset()

    def next(self, i=None):
        if i is None:
            i = self.active
        evaluation = self.evaluate_condition(i)
        if self.lines[i] is If:
            self.if_evaluations.append(evaluation)
        i = self.line_transitions[i][evaluation]
        if i >= len(self.lines):
            return None
        if self.lines[i] is Subtask:
            return i
        return self.next(i)

    def initial(self):
        if self.lines[0] is Subtask:
            return 0
        return self.next(i=0)

    def evaluate_condition(self, i=None):
        if i is None:
            i = self.active
        if self.lines[i] is Else:
            return not self.if_evaluations.pop()
        return self._evaluate_condition(i)

    @abc.abstractmethod
    def _evaluate_condition(self, i=None):
        pass
