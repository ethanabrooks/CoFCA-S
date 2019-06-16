import operator
from functools import reduce

import torch
from torch import nn as nn
import torch.jit

from ppo.utils import broadcast_3d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(torch.jit.ScriptModule):
    def __init__(self, dim=-1):
        self.dim = dim
        super().__init__()

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Reshape(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(*self.shape)


class Broadcast3d(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return broadcast_3d(inputs, self.shape)


class Product(torch.jit.ScriptModule):
    @staticmethod
    def forward(inputs):
        return reduce(operator.mul, inputs, 1)


class Parallel(torch.jit.ScriptModule):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, inputs):
        return tuple([m(x) for m, x in zip(self.modules, inputs)])


