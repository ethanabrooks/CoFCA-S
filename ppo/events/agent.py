import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

from ppo.agent import AgentValues, NNBase
import ppo.agent
from ppo.distributions import FixedCategorical, Categorical

# noinspection PyMissingConstructor
from ppo.events.recurrence import Recurrence, RecurrentState


class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, entropy_coef, **kwargs):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        self.recurrent_module = self.build_recurrent_module(**kwargs)

    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        rm = self.recurrent_module
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        hx = RecurrentState(*rm.parse_hidden(all_hxs))
        dist = FixedCategorical(hx.a_probs)
        entropy = dist.entropy().mean()
        return AgentValues(
            value=hx.v,
            action=hx.a,
            action_log_probs=dist.log_probs(hx.a).mean(),
            aux_loss=self.entropy_coef * entropy,
            dist=dist,
            rnn_hxs=last_hx,
            log=dict(entropy=entropy),
        )

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(last_hx).v

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            y = torch.cat([x, action.float()], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True
