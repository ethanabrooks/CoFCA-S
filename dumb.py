#! /usr/bin/env python
from dataclasses import dataclass

import torch


class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.gru = torch.nn.GRU(4, 4)

    def forward(self, x):
        return self.gru(x)


scripted_gate = torch.jit.script(MyDecisionGate())


class ScriptGRU(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ScriptGRU, self).__init__()
        self.gru = torch.nn.GRU(*args, **kwargs)

    def forward(self, x):
        return self.gru(x)


my_cell = ScriptGRU(4, 4)
traced_cell = torch.jit.script(my_cell)
print(traced_cell(torch.randn(1, 2, 4)))
their_cell = torch.nn.GRU(4, 4)
traced_cell = torch.jit.script(their_cell)
print(traced_cell(torch.randn(1, 2, 4)))
# gru = ScriptGRU(4, 4)
# print(gru(torch.randn(1, 2, 4)))
