from abc import ABC


class Constraint:
    def satisfied(self, columns):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError


class SideBySide(Constraint, ABC):
    def __init__(self, left: int, right: int):
        self.right = right
        self.left = left

    def satisfied(self, columns):
        left, right = [
            next(i for i, c in enumerate(columns) if x in c)
            for x in [self.left, self.right]
        ]
        return left + 1 == right

    def __str__(self):
        return f"{self.left} left of {self.right}"


class Left(SideBySide):
    def list(self):
        return [self.left, 0, self.right]


class Right(SideBySide):
    def list(self):
        return [self.right, 1, self.left]


class Stacked(Constraint, ABC):
    def __init__(self, top: int, bottom: int):
        self.top = top
        self.bottom = bottom

    def satisfied(self, columns):
        try:
            column = next(c for c in columns if self.top in c and self.bottom in c)
        except StopIteration:
            return False
        return column.index(self.top) == column.index(self.bottom) + 1

    def __str__(self):
        return f"{self.top} above {self.bottom}"


class Above(Stacked):
    def list(self):
        return [self.top, 2, self.bottom]


class Below(Stacked):
    def list(self):
        return [self.bottom, 3, self.top]
