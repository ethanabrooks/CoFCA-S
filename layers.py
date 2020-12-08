import math
from functools import reduce
import operator
from typing import Union

import numpy as np
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

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


class MultiEmbeddingBag(nn.Module):
    def __init__(self, nvec: Union[np.ndarray, torch.Tensor], **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=nvec.sum(), **kwargs)
        self.register_buffer(
            "offset",
            F.pad(torch.tensor(nvec[:-1]).cumsum(0), [1, 0]),
        )

    def forward(self, inputs):
        return self.embedding(self.offset + inputs).sum(-2)


class IntEncoding(nn.Module):
    def __init__(self, d_model: int):
        self.d_model = d_model
        nn.Module.__init__(self)
        self.register_buffer(
            "div_term",
            torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            ),
        )

    def forward(self, x):
        shape = x.shape
        div_term = self.div_term.view(*(1 for _ in shape), -1)
        x = x.unsqueeze(-1)
        sins = torch.sin(x * div_term)
        coss = torch.cos(x * div_term)
        return torch.stack([sins, coss], dim=-1).view(*shape, self.d_model)
