import abc
from abc import ABC
from dataclasses import dataclass


class Line(abc.ABC):
    @abc.abstractmethod
    def to_tuple(self):
        raise NotImplementedError


class Else(Line, ABC):
    def __str__(self):
        return "else:"


class EndIf(Line, ABC):
    def __str__(self):
        return "endif"


class EndWhile(Line, ABC):
    def __str__(self):
        return "endwhile"


@dataclass
class If(Line, ABC):
    object: int
    object_types: list

    def __str__(self):
        return f"if {self.object_types[self.object]}:"


@dataclass
class While(Line, ABC):
    object: int
    object_types: list

    def __str__(self):
        return f"while {self.object_types[self.object]}:"


@dataclass
class Subtask(Line, ABC):
    pass
