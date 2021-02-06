import typing
from dataclasses import dataclass, astuple
from typing import Union, Generator, Generic

import numpy as np
import torch


@dataclass(frozen=True)
class RawAction:
    delta: Union[np.ndarray, torch.Tensor, X]
    gate: Union[np.ndarray, torch.Tensor, X]
    pointer: Union[np.ndarray, torch.Tensor, X]
    extrinsic: Union[np.ndarray, torch.Tensor, X]

    @staticmethod
    def parse(*xs) -> "RawAction":
        delta, gate, ptr, *extrinsic = xs
        if extrinsic == [None]:
            extrinsic = None
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
