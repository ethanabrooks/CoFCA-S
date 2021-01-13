from dataclasses import dataclass
import torch

import our_agent


@dataclass
class Agent(our_agent.Agent):
    def __hash__(self):
        return self.hash()

    def build_beta(self):
        return None

    def build_d_gate(self):
        return None

    @staticmethod
    def build_m(M, R, p):
        return M.view(M.size(0), -1)

    def build_P(self, *args, **kwargs):
        return None

    def build_task_encoder(self):
        return

    def build_upsilon(self):
        return None

    def get_dg(self, can_open_gate, ones, z):
        return torch.ones_like(ones), None

    def get_delta(self, P, dg, line_mask, ones, z):
        return torch.ones_like(ones) * self.nl, None

    def get_gru_in_size(self):
        return self.action_embed_size + self.nl * self.instruction_embed_size
