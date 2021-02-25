import typing
from dataclasses import dataclass, astuple
from typing import Generator, Generic, Any


@dataclass(frozen=True)
class RawAction:
    delta: Any
    gate: Any
    pointer: Any
    extrinsic: Any

    @staticmethod
    def parse(delta, gate, ptr, extrinsic) -> "RawAction":
        return RawAction(delta, gate, ptr, extrinsic)

    def flatten(self) -> Generator[any, None, None]:
        yield from astuple(self)


X = typing.TypeVar("X")


@dataclass
class RecurrentState(Generic[X]):
    a: X
    d: X
    h: X
    dg: X
    p: X
    v: X
    a_probs: X
    d_probs: X
    dg_probs: X


@dataclass(frozen=True)
class Obs:
    action_mask: Any
    destroyed_unit: Any
    gate_openers: Any
    instruction_mask: Any
    instructions: Any
    obs: Any
    partial_action: Any
    pointer: Any
    resources: Any
