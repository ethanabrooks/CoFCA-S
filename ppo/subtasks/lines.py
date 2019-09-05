import abc
import enum
from abc import ABC
from dataclasses import dataclass
from enum import Enum


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


class Lines(Enum):
    If = enum.auto
    Else = enum.auto
    EndIf = enum.auto
    While = enum.auto
    EndWhile = enum.auto
    Subtask = enum.auto


If = Lines.If
Else = Lines.Else
EndIf = Lines.EndIf
While = Lines.While
EndWhile = Lines.EndWhile
Subtask = Lines.Subtask


class LineStates(Enum):
    initial = enum.auto
    following_if = enum.auto
    inside_if = enum.auto
    following_else = enum.auto
    inside_else = enum.auto
    following_while = enum.auto
    inside_while = enum.auto


initial = LineStates.initial
following_if = LineStates.following_if
inside_if = LineStates.inside_if
following_else = LineStates.following_else
inside_else = LineStates.iniside_else
following_while = LineStates.following_while
inside_while = LineStates.inside_while
