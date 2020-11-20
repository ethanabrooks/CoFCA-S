#! /usr/bin/env python
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        print(torch.round(100 * torch.sin(position * div_term))[:10])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


pe = PositionalEncoding(10)

d_model = 10


@dataclass
class IntEncoding(nn.Module):
    d_model: int

    def __post_init__(self):
        self.div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

    def forward(self, x):
        shape = x.shape
        div_term = self.div_term.view(*(1 for _ in shape), -1)
        x = x.unsqueeze(-1)
        sins = torch.sin(x * div_term)
        coss = torch.cos(x * div_term)
        return torch.stack([sins, coss], dim=-1).view(*shape, self.d_model)


enc = IntEncoding(10)
x = torch.arange(2 * 3).view(3, 2)
print(torch.round(100 * enc.forward(x)).shape)
print(torch.round(100 * enc.forward(x)))
