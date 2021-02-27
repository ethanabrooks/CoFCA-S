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

    def forward_gru(self, destroyed_unit, embedded_action, m, masks, rnn_hxs, x):
        y = torch.cat([x, destroyed_unit, embedded_action, m], dim=-1)
        z, rnn_hxs = self._forward_gru(y, rnn_hxs, masks, self.gru)
        return z, rnn_hxs

    @property
    def gru_in_size(self):
        return (
            self.conv_hidden_size
            + self.action_embed_size
            + self.destroyed_unit_embed_size
            + self.instruction_embed_size
        )
