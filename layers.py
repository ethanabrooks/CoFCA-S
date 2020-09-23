from functools import reduce
import operator

import torch
from torch import nn as nn
import torch.jit

from utils import broadcast3d


class CumSum(torch.jit.ScriptModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def forward(self, inputs):
        return torch.cumsum(inputs, **self.kwargs)


class Squash(nn.Module):
    def forward(self, x):
        y = x ** 3
        return torch.clamp(y, min=0) / (1 + y.abs())


class Flatten(nn.Module):
    def __init__(self, out_size=None):
        super().__init__()
        self.out_size = out_size

    def forward(self, x):
        return x.view(x.size(0), -1)


class Print(nn.Module):
    def __init__(self, f=None):
        self.f = f
        super().__init__()

    def forward(self, x):
        if self.f is None:
            print(x)
        else:
            print(self.f(x))
        return x


class Log(nn.Module):
    def forward(self, x):
        return torch.log(x)


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Sum(nn.Module):
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        super().__init__()

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            return sum(inputs)
        else:
            return torch.sum(inputs, **self.kwargs)


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


def wrap_parameter(x, requires_grad):
    if isinstance(x, torch.Tensor):
        return nn.Parameter(x, requires_grad=requires_grad)
    else:
        return x


class Plus(torch.jit.ScriptModule):
    def __init__(self, x, requires_grad=False):
        super().__init__()
        self.x = wrap_parameter(x, requires_grad)

    def forward(self, inputs):
        return self.x + inputs


class Times(torch.jit.ScriptModule):
    def __init__(self, x, requires_grad=False):
        super().__init__()
        self.x = wrap_parameter(x, requires_grad)

    def forward(self, inputs):
        return self.x * inputs
