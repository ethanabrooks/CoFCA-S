class Line:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return f"{self.__class__.__name__} {self.id}"

    @property
    def terminates(self):
        return None

    @property
    def depth_change(self) -> int:
        return 0

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))


class If(Line):
    @property
    def depth_change(self) -> int:
        return 1


class Else(Line):
    pass


class EndIf(Line):
    @property
    def terminates(self):
        return If

    @property
    def depth_change(self) -> int:
        return -1


class While(Line):
    @property
    def depth_change(self) -> int:
        return 1


class EndWhile(Line):
    @property
    def terminates(self):
        return While

    @property
    def depth_change(self) -> int:
        return -1


class Loop(Line):
    @property
    def depth_change(self) -> int:
        return 1


class EndLoop(Line):
    @property
    def depth_change(self) -> int:
        return -1


class Subtask(Line):
    pass


class Padding(Line):
    pass
