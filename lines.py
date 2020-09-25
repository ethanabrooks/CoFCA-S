from abc import ABC
from typing import List, Type, Generator

# noinspection PyShadowingBuiltins
from numpy.random.mtrand import RandomState


def sample(random, _min, _max, p=0.5):
    return min(_min + random.geometric(p) - 1, _max)


class Line:
    types = None
    required_lines = 0
    required_depth = 0
    depth_change = None

    def __init__(self, id):
        self.id = id

    def __str__(self):
        return f"{self.__class__.__name__} {self.id}"

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))

    @staticmethod
    def generate_types(n, remaining_depth, random, legal_lines):
        # type: (int, int, RandomState, List[Type[Line]]) -> Generator[Type[Line]]
        assert Subtask in legal_lines
        if n == 0:
            return
        _legal_lines = [
            l
            for l in legal_lines
            if l.required_lines <= n and l.required_depth <= remaining_depth
        ]
        assert Subtask in _legal_lines
        if _legal_lines:
            line = random.choice(_legal_lines)
            m = sample(random, line.required_lines, n)
            # line.check(m, remaining_depth)
            yield from line.generate_types(
                m, remaining_depth, random=random, legal_lines=legal_lines
            )
            n -= m
        yield from Line.generate_types(
            n, remaining_depth, random=random, legal_lines=legal_lines
        )

    @staticmethod
    def transitions(line_index, previous_condition):
        raise NotImplementedError


class If(Line):
    required_lines = 3
    required_depth = 1
    depth_change = 0, 1

    @staticmethod
    def generate_types(n: int, remaining_depth: int, legal_lines: list, **kwargs):
        yield If
        yield from Line.generate_types(
            n - 2, remaining_depth - 1, **kwargs, legal_lines=legal_lines
        )
        yield EndIf

    def transition(self, prev, current):
        raise NotImplementedError

    @staticmethod
    def transitions(line_index, previous_condition):
        previous_condition.append(line_index)
        yield from ()


class Else(Line):
    required_lines = 5
    required_depth = 1
    depth_change = -1, 1

    @staticmethod
    def generate_types(n, remaining_depth, random, **kwargs):
        assert n >= Else.required_lines
        n -= 3
        assert n >= 2
        m = sample(random, 1, n - 1)
        yield If
        yield from Line.generate_types(m, remaining_depth - 1, random, **kwargs)
        yield Else
        yield from Line.generate_types(n - m, remaining_depth - 1, random, **kwargs)
        yield EndIf

    @staticmethod
    def transitions(line_index, previous_condition):
        prev = previous_condition[-1]
        yield prev, line_index  # False: If -> Else
        yield prev, prev + 1  # True: If -> If + 1
        previous_condition[-1] = line_index


class EndIf(Line):
    depth_change = -1, 0

    @property
    def terminates(self):
        return If

    @staticmethod
    def generate_types(*args, **kwargs):
        raise RuntimeError

    @staticmethod
    def transitions(line_index, previous_condition):
        prev = previous_condition[-1]
        yield prev, line_index  # False: If/Else -> EndIf
        yield prev, prev + 1  # True: If/Else -> If/Else + 1

        yield line_index, line_index + 1  # False: If/Else -> If/Else + 1
        yield line_index, line_index + 1  # True: If/Else -> If/Else + 1


class Loop(Line):
    required_lines = 3
    required_depth = 1
    depth_change = 0, 1

    @classmethod
    def generate_types(cls, n: int, remaining_depth: int, **kwargs):
        yield cls
        yield from Line.generate_types(n - 2, remaining_depth - 1, **kwargs)
        yield cls.end_loop()

    @staticmethod
    def transitions(line_index, previous_condition):
        previous_condition.append(line_index)
        yield from ()

    @classmethod
    def end_loop(cls):
        raise NotImplementedError


class EndLoop(Line, ABC):
    depth_change = -1, 0

    @staticmethod
    def generate_types(*args, **kwargs):
        raise RuntimeError

    @staticmethod
    def transitions(line_index, previous_conditions):
        prev = previous_conditions[-1]
        # While
        yield prev, line_index + 1  # False: While -> EndWhile + 1
        yield prev, prev + 1  # True: While -> While + 1
        # EndWhile
        yield line_index, prev  # False: EndWhile -> While
        yield line_index, prev  # True: EndWhile -> While


class While(Loop):
    @classmethod
    def end_loop(cls):
        return EndWhile


class EndWhile(EndLoop):
    pass


class For(Loop):
    @classmethod
    def end_loop(cls):
        return EndLoop


class EndFor(EndLoop):
    pass


class Subtask(Line):
    required_lines = 1
    required_depth = 0
    depth_change = 0, 0

    @staticmethod
    def generate_types(n: int, *args, **kwargs):
        yield Subtask
        yield from Line.generate_types(n - 1, *args, **kwargs)

    @staticmethod
    def transitions(line_index, _):
        yield line_index, line_index + 1
        yield line_index, line_index + 1


class SelfLoop(Subtask):
    required_lines = 1
    required_depth = 0
    depth_change = 0, 0

    @classmethod
    def generate_types(cls, n: int, *args, **kwargs):
        yield cls
        yield from Line.generate_types(n - 1, *args, **kwargs)

    @staticmethod
    def transitions(line_index, _):
        yield line_index, line_index  # False: do not advance
        yield line_index, line_index + 1  # True: advance by 1


class Some(SelfLoop):
    pass


class Most(SelfLoop):
    pass


class ForAWhile(Loop):
    @classmethod
    def end_loop(cls):
        return EndForAWhile


class EndForAWhile(EndLoop):
    pass


class EnoughToBuy(Loop):
    @classmethod
    def end_loop(cls):
        return EndEnoughToBuy


class EndEnoughToBuy(EndLoop):
    pass


class Padding(Line, ABC):
    pass


# Line.types = [Subtask, If, Else, EndIf, While, EndWhile, Loop, EndLoop, Padding]
Line.types = [
    Subtask,
    If,
    Else,
    EndIf,
    While,
    EndWhile,
    For,
    EndFor,
    Padding,
    Some,
    Most,
    ForAWhile,
    EndForAWhile,
    EnoughToBuy,
    EndEnoughToBuy,
]
