from functools import reduce
import operator

import torch
from torch import nn as nn
import torch.jit

from ppo.utils import broadcast3d


class CumSum(torch.jit.ScriptModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def forward(self, inputs):
        return torch.cumsum(inputs, **self.kwargs)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Sum(nn.Module):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def forward(self, x):
        return torch.sum(x, **self.kwargs)


class ShallowCopy(nn.Module):
    def __init__(self, n: int):
        self.n = n
        super().__init__()

    def forward(self, x):
        return (x,) * self.n


class Concat(torch.jit.ScriptModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def forward(self, inputs):
        return torch.cat(inputs, **self.kwargs)


class Reshape(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(inputs.size(0), *self.shape)


class Broadcast3d(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return broadcast3d(inputs, self.shape)


class Product(torch.jit.ScriptModule):
    @staticmethod
    def forward(inputs):
        return reduce(operator.mul, inputs, 1)


class Parallel(torch.jit.ScriptModule):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, inputs):
        return tuple([m(x) for m, x in zip(self.module_list, inputs)])
