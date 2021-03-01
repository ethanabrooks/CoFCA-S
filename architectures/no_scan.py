from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

from architectures import ours


@dataclass
class Agent(ours.Agent):
    def __hash__(self):
        return self.hash()

    def build_beta(self):
        return nn.Sequential(
            self.init_(
                nn.Linear(
                    self.z_size,
                    self.delta_size,
                )
            )
        )

    @property
    def max_backward_jump(self):
        return self.eval_lines

    @property
    def max_forward_jump(self):
        return self.eval_lines - 1

    def build_upsilon(self):
        return None

    def get_delta_probs(self, G, P, z):
        return torch.softmax(self.beta(z), dim=-1)

    def get_G_g(self, rolled):
        _, g = super().get_G_g(rolled)
        return g, g

    def get_instruction_mask(self, N, instruction_mask):
        instruction_mask = super().get_instruction_mask(N, instruction_mask)
        instruction_mask = F.pad(
            instruction_mask,
            (0, self.delta_size - instruction_mask.size(-1)),
            value=1,
        )
        return instruction_mask

    def get_P(self, *args, **kwargs):
        return None

    def get_g(self, G, _, __):
        N = G.size(0)
        return G.reshape(N, 2 * self.instruction_embed_size)
