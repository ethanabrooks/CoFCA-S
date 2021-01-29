from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

import cofi_s


@dataclass
class Agent(cofi_s.Agent):
    def __hash__(self):
        return self.hash()

    def build_beta(self):
        return nn.Sequential(
            self.init_(
                nn.Linear(
                    self.instruction_embed_size * 2,  # biGRU
                    self.delta_size,
                )
            )
        )

    def get_critic_input(self, G, R, p, z1, zc):
        if self.globalized_critic:
            zc = torch.cat([z1, G], dim=-1)
            if self.add_layer:
                zc = self.eta(zc)
        return zc

    @property
    def max_backward_jump(self):
        return self.eval_lines

    @property
    def max_forward_jump(self):
        return self.eval_lines - 1

    def build_upsilon(self):
        return None

    def get_delta_probs(self, G, P, z):
        return torch.softmax(self.beta(G), dim=-1)

    def get_G(self, rolled):
        N = rolled.size(0)
        _, G = self.encode_G(rolled)
        return G.transpose(0, 1).reshape(N, 2 * self.instruction_embed_size)

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
