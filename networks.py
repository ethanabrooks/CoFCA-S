import math
from typing import List

import torch
from torch import nn as nn
from torch.nn import functional as F


class MultiEmbeddingBag(torch.jit.ScriptModule):
    def __init__(self, nvec: List[int], **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=sum(nvec), **kwargs)
        self.register_buffer(
            "offset",
            F.pad(torch.tensor(nvec[:-1]).cumsum(0), [1, 0]),
        )

    def forward(self, inputs):
        return self.embedding(self.offset + inputs).sum(-2)


class IntEncoding(torch.jit.ScriptModule):
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
        div_term = self.div_term
        x = x.unsqueeze(-1)
        while len(div_term.shape) < len(x.shape):
            div_term = div_term.unsqueeze(0)
        sins = torch.sin(x * div_term)
        coss = torch.cos(x * div_term)
        return torch.stack([sins, coss], dim=-1)


class GRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__()
        self.gru = torch.nn.GRU(*args, **kwargs)

    def forward(self, x):
        return self.gru(x)
