from typing import List, Type, Generator

import numpy as np


# noinspection PyShadowingBuiltins
from numpy.random.mtrand import RandomState


def split(n: int, p: float, random: RandomState):
    m = random.binomial(n, p)
    return m, n - m


class Line:
    types = None
    legal_next_lines = None
    expression_starts = None
    termination = None
    control_flow_lines = None
    required_lines = 0
    required_depth = 0

    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.__class__.__name__

    @property
    def depth_change(self) -> int:
        return 0

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))

    @staticmethod
    def generate_lines(n, remaining_depth, random, legal_lines):
        # type: (int, int, RandomState, List[Type[Line]]) -> Generator[Type[Line]]
        if n == 0:
            return
        m = 1 + random.binomial(n - 1, 0.5)
        _legal_lines = [
            l
            for l in legal_lines
            if l.required_lines <= m and l.required_depth <= remaining_depth
        ]
        if _legal_lines:
            line = random.choice(_legal_lines)
            # line.check(m, remaining_depth)
            yield from line.generate_lines(
                m, remaining_depth, random=random, legal_lines=legal_lines
            )
            n -= m
        yield from Line.generate_lines(
            n, remaining_depth, random=random, legal_lines=legal_lines
        )


class If(Line):
    required_lines = 3
    required_depth = 1

    @property
    def depth_change(self) -> int:
        return 1

    @staticmethod
    def generate_lines(n: int, remaining_depth: int, legal_lines: list, **kwargs):
        yield If
        yield from Line.generate_lines(
            n - 2, remaining_depth - 1, **kwargs, legal_lines=legal_lines + [Else]
        )
        yield EndIf


class Else(Line):
    required_lines = 2
    required_depth = 1

    @staticmethod
    def generate_lines(n, remaining_depth, **kwargs):
        yield Else
        yield from Line.generate_lines(n - 1, remaining_depth - 1, **kwargs)


class EndIf(Line):
    @property
    def terminates(self):
        return If

    @property
    def depth_change(self) -> int:
        return -1

    @staticmethod
    def generate_lines(*args, **kwargs):
        raise RuntimeError


class While(Line):
    required_lines = 3
    required_depth = 1

    @property
    def depth_change(self) -> int:
        return 1

    @staticmethod
    def generate_lines(n: int, remaining_depth: int, **kwargs):
        yield While
        yield from Line.generate_lines(n - 2, remaining_depth - 1, **kwargs)
        yield EndWhile


class EndWhile(Line):
    @property
    def terminates(self):
        return While

    @property
    def depth_change(self) -> int:
        return -1

    @staticmethod
    def generate_lines(*args, **kwargs):
        raise RuntimeError


class Loop(Line):
    required_lines = 3
    required_depth = 1

    @property
    def depth_change(self) -> int:
        return 1

    def __str__(self):
        return f"{self.__class__.__name__} {self.id}"

    @staticmethod
    def generate_lines(n: int, remaining_depth: int, **kwargs):
        yield Loop
        yield from Line.generate_lines(n - 2, remaining_depth - 1, **kwargs)
        yield EndLoop


class EndLoop(Line):
    @property
    def depth_change(self) -> int:
        return -1

    @staticmethod
    def generate_lines(*args, **kwargs):
        raise RuntimeError


class Subtask(Line):
    required_lines = 1
    required_depth = 0

    def __str__(self):
        return f"{self.__class__.__name__} {self.id}"

    @staticmethod
    def generate_lines(n: int, *args, **kwargs):
        yield Subtask
        yield from Line.generate_lines(n - 1, *args, **kwargs)


class Padding(Line):
    pass


Line.types = {Subtask, If, Else, EndIf, While, EndWhile, Loop, EndLoop}
Line.legal_next_lines = {Subtask}
Line.control_flow_lines = {Subtask}
Line.expression_starts = [Subtask, If, While, Loop]
