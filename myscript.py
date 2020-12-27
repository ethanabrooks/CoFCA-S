from typing import Optional
from dataclasses import dataclass, MISSING


@dataclass
class A:
    x: Optional[int] = None


class B(A):
    x: Optional[int] = MISSING


print(A())
print(B())
