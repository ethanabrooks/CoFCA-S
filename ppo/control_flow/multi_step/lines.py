import functools
from enum import Enum
from typing import List, Type, Generator, Tuple

# noinspection PyShadowingBuiltins
from adt import adt, Case
from numpy.random.mtrand import RandomState


import ppo.control_flow.lines as single_step


@adt
class Terrain:
    BRIDGE = Case
    AGENT = Case


@adt
class Item:
    WOOD = Case
    GOLD = Case
    IRON = Case
    MERCHANT = Case


@adt
class Subtask:
    MINE = Case[Item]
    SELL = Case[Item]
    GOTO = Case[Item]


@adt
class Line(single_step.Line):
    SUBTASK = Case[Subtask]
    IF = Case[Item]
    ELSE = Case
    ENDIF = Case
    WHILE = Case[Item]
    ENDWHILE = Case
    LOOP = Case[int]
    ENDLOOP = Case
    PADDING = Case


class Subtask(Line, single_step.Subtask):
    pass


class If(Line, single_step.If):
    pass


class Else(Line, single_step.Else):
    pass


class EndIf(Line, single_step.EndIf):
    pass


class While(Line, single_step.While):
    pass


class EndWhile(Line, single_step.EndWhile):
    pass


class Loop(Line, single_step.Loop):
    pass


class EndLoop(Line, single_step.EndLoop):
    pass


class Padding(Line, single_step.Padding):
    pass
