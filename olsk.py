from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

import cofca
import cofca_s


@dataclass
class Agent(cofca.Agent):
    def __post_init__(self):
        super().__post_init__()

    def __hash__(self):
        return self.hash()

    @property
    def max_backward_jump(self):
        return 1

    @property
    def max_forward_jump(self):
        return 1

    def build_upsilon(self):
        return None
