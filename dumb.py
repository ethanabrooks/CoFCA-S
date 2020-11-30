#! /usr/bin/env python
from collections import namedtuple
from dataclasses import dataclass

import torch

from torch.distributions import Categorical


class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x


X = namedtuple("X", "a b")


class Y(torch.jit.ScriptModule):
    a: torch.Tensor
    b: torch.Tensor


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dist = Categorical(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.gru = torch.nn.GRU(4, 4)

    def forward(self, x):
        a, b = torch.split(x, [2, 2], dim=-1)
        return self.dist.sample()


scripted_gate = torch.jit.script(MyDecisionGate())


my_cell = Categorical(torch.tensor([0.0, 1.0]))
traced_cell = torch.jit.script(my_cell)
print(traced_cell.sample())
# gru = ScriptGRU(4, 4)
# print(gru(torch.randn(1, 2, 4)))
