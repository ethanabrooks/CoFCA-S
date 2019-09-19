from abc import ABC
import itertools


class Constraint:
    def satisfied(self, columns):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError


class SideBySide(Constraint, ABC):
    def __init__(self, left: int, right: int):
        self.right = right or 0
        self.left = left or 0

    def satisfied(self, columns):
        for left_column, right_column in zip(columns, columns[1:]):
            for left, right in zip(left_column, right_column):
                if self.left == left and self.right == right:
                    return True
        return False


class Left(SideBySide):
    def list(self):
        return [self.left, 0, self.right]

    def __str__(self):
        return f"{self.left} left of {self.right}"


class Right(SideBySide):
    def list(self):
        return [self.right, 1, self.left]

    def __str__(self):
        return f"{self.right} right of {self.left}"


class Stacked(Constraint, ABC):
    def __init__(self, top: int, bottom: int):
        self.top = top or 0
        self.bottom = bottom or 0

    def satisfied(self, columns):
        for column in columns:
            for bottom, top in zip(column, column[1:]):
                if bottom == self.bottom and top == self.top:
                    return True
        return False


class Above(Stacked):
    def list(self):
        return [self.top, 1, self.bottom]

    def __str__(self):
        return f"{self.top} above {self.bottom}"


class Below(Stacked):
    def list(self):
        return [self.bottom, 3, self.top]

    def __str__(self):
        return f"{self.bottom} below {self.top}"
