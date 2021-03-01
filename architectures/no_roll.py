from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

from architectures import ours


@dataclass
class Agent(ours.Agent):
    def __hash__(self):
        return self.hash()

    def get_rolled(self, M, R, p):
        one_hot = torch.zeros(M.size(0), M.size(1), 1, device=M.device)
        one_hot[R, p] = 1
        return torch.cat([M, one_hot], dim=-1)

    @property
    def rolled_size(self):
        return self.instruction_embed_size + 1

    @property
    def z_size(self):
        return self.s_size + 2 * self.num_gru_layers * (self.instruction_embed_size + 1)
