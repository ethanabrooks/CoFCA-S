from dataclasses import dataclass


class Z:
    def __init__(self, v):
        self.v = v


@dataclass
class A(Z):
    a: int = 1

    def __post_init__(self):
        Z.__init__(self, 1)


@dataclass
class B(Z):
    b: int = 2


@dataclass
class C(A, B):
    pass


print(C().v)
