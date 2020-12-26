from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class A:
    @property
    def a(self):
        return None


@dataclass
class B(A):
    a = 1


print(A().a)
print(B().a)
