from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

import cofi
import cofi_s


@dataclass
class Agent(cofi.Agent):
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

    @property
    def beta_in(self):
        return self.z_size

    def get_delta_probs(self, G, P, z):
        return torch.softmax(self.beta(z), dim=-1)
