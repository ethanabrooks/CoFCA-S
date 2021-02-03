from dataclasses import dataclass
import torch
import torch.nn as nn

import cofi_s


@dataclass
class Agent(cofi_s.Agent):
    def __post_init__(self):
        self.globalized_critic = False
        self.feed_m_to_gru = False
        super().__post_init__()

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
        return torch.ones_like(ones) * self.instruction_length, None

    def get_delta_probs(self, G, P, z):
        return None

    def get_rolled(self, M, R, p):
        return None

    def get_G_g(self, rolled):
        return None

    def build_g_gru(self):
        return None

    def get_g(self, G, R, p):
        return None

    def update_hxs(
        self, embedded_action, destroyed_unit, action_rnn_hxs, g, g_rnn_hxs, masks
    ):
        ha, action_rnn_hxs = self._forward_gru(
            embedded_action, action_rnn_hxs, masks, gru=self.action_gru
        )
        return ha, action_rnn_hxs, None, None

    def split_rnn_hxs(self, rnn_hxs):
        return rnn_hxs, None

    @staticmethod
    def combine_rnn_hxs(action_rnn_hxs, g_rnn_hxs):
        return action_rnn_hxs

    @property
    def recurrent_hidden_state_size(self):
        return self.hidden_size

    def get_zg(self, z, hg, za):
        return za

    @property
    def za_size(self):
        return self.z_size + 2 * self.instruction_embed_size

    @property
    def zg_size(self):
        return self.z_size + 2 * self.instruction_embed_size
