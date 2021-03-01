from dataclasses import dataclass

from architectures import no_scan
import torch
import torch.nn as nn


@dataclass
class Agent(no_scan.Agent):
    def __post_init__(self):
        super().__post_init__()
        #
        # self.olsk_gru = nn.GRU(
        #     self.gru_in_size,
        #     self.hidden_size,
        # )

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

    def print(*args, **kwargs):
        pass

    def forward_gru(self, s, rnn_hxs, masks):
        h, rnn_hxs = self._forward_gru(s, rnn_hxs, masks, self.gru)
        return h, rnn_hxs

    def get_z(self, h, s, g):
        g = g.reshape(g.size(0), 2 * self.num_gru_layers * self.rolled_size)
        return torch.cat([s, h, g], dim=-1)

    @property
    def z_size(self):
        return (
            self.h_size
            + self.s_size
            + 2 * self.num_gru_layers * self.instruction_embed_size
        )

    @property
    def f_in_size(self):
        return (
            self.conv_hidden_size
            + self.action_embed_size
            + self.destroyed_unit_embed_size
            + self.instruction_embed_size
        )
