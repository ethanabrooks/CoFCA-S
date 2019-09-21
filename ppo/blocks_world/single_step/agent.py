from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ppo
from ppo.agent import AgentValues, NNBase
from ppo.distributions import FixedCategorical

# noinspection PyMissingConstructor

AgentValues = namedtuple(
    "AgentValues", "value action action_log_probs aux_loss rnn_hxs log dist"
)


class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, entropy_coef, model_loss_coef, recurrence):
        nn.Module.__init__(self)
        self.model_loss_coef = model_loss_coef
        self.entropy_coef = entropy_coef
        self.recurrent_module = recurrence

    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        rm = self.recurrent_module
        hx = rm.parse_hidden(all_hxs)
        dist = FixedCategorical(hx.p)
        action_log_probs = dist.log_probs(hx.a)
        aux_loss = -self.entropy_coef * dist.entropy()
        return AgentValues(
            value=hx.v,
            action=hx.a,
            action_log_probs=action_log_probs,
            aux_loss=aux_loss.mean(),
            dist=dist,
            rnn_hxs=last_hx,
            log=dict(entropy=(dist.entropy())),
        )

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            y = torch.cat([x, action.float()], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    def get_value(self, inputs, rnn_hxs, masks):
        all_hxs, last_hx = self._forward_gru(
            inputs.view(inputs.size(0), -1), rnn_hxs, masks
        )
        return self.recurrent_module.parse_hidden(last_hx).v
