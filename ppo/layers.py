import torch
import torch.jit
from torch import nn as nn

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