import functools
from enum import Enum
from typing import List, Type, Generator, Tuple

# noinspection PyShadowingBuiltins
from numpy.random.mtrand import RandomState


import ppo.control_flow.lines as single_step


class Line(single_step.Line):
    pass


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
