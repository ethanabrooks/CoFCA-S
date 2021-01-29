from dataclasses import dataclass
import torch
import torch.nn as nn

import cofi_s


@dataclass
class Agent(cofi_s.Agent):
    def __hash__(self):
        return self.hash()

    def build_beta(self):
        return None

    def build_gate(self):
        return None

    def build_encode_G(self):
        return nn.GRU(
            self.instruction_embed_size,
            self.instruction_embed_size,
            bidirectional=True,
            batch_first=True,
        )

    def build_m(self, M, R, p):
        _, m = self.encode_G(M)
        return m.transpose(0, 1).reshape(m.size(1), 2 * m.size(2))

    def get_P(self, *args, **kwargs):
        return None

    def build_task_encoder(self):
        return

    def build_upsilon(self):
        return None

    def get_gate(self, can_open_gate, ones, z):
        return torch.ones_like(ones), None

    def get_delta(self, delta_probs, dg, line_mask, ones):
        return torch.ones_like(ones) * self.nl, None

    def get_delta_probs(self, G, P, z):
        return None

    def get_gru_in_size(self):
        assert self.feed_m_to_gru is False
        return self.action_embed_size

    def get_rolled(self, M, R, p, z1):
        return None

    def get_G(self, rolled):
        return None

    @property
    def zeta_input_size(self):
        return self.z1_size + 2 * self.instruction_embed_size
