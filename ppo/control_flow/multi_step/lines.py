# noinspection PyShadowingBuiltins


import lines as single_step


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
