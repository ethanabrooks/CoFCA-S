class Line:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return f"{self.__class__.__name__} {self.id}"


class If(Line):
    pass


class Else(Line):
    pass


class EndIf(Line):
    pass


class While(Line):
    pass


class EndWhile(Line):
    pass


class Subtask(Line):
    pass


class Padding(Line):
    pass
