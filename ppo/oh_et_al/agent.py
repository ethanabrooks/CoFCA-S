import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import AgentValues, NNBase

# noinspection PyMissingConstructor
from ppo.distributions import FixedCategorical


class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, entropy_coef, recurrence):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        self.recurrent_module = recurrence

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        rm = self.recurrent_module
        hx = rm.parse_hidden(all_hxs)
        a_dist = FixedCategorical(hx.a_probs)
        b_dist = FixedCategorical(hx.b_probs)
        action_log_probs = torch.cat(
            [a_dist.log_probs(hx.a), b_dist.log_probs(hx.b)], dim=-1
        )
        entropy = (a_dist.entropy() + b_dist.entropy()).mean()
        return AgentValues(
            value=torch.cat([hx.v0, hx.v1], dim=-1),
            action=torch.cat([hx.a, hx.b], dim=-1),
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=None,
            rnn_hxs=last_hx,
            log=dict(entropy=entropy),
        )

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            y = torch.cat([x, action.float()], dim=-1)
        return super()._forward_gru(y, hxs, masks[:, :1])

    def get_value(self, inputs, rnn_hxs, masks):
        all_hxs, last_hx = self._forward_gru(
            inputs.view(inputs.size(0), -1), rnn_hxs, masks
        )
        hx = self.recurrent_module.parse_hidden(last_hx)
        return torch.cat([hx.v0, hx.v1], dim=-1)