from dataclasses import astuple

import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

import agents
import ours
from agents import AgentOutputs, NNBase
from distribution_modules import FixedCategorical
from distributions import JointDistribution


class Agent(agents.Agent, NNBase):
    def __init__(
        self,
        entropy_coef: float,
        gate_coef: float,
        **network_args,
    ):
        nn.Module.__init__(self)
        self.gate_coef = gate_coef
        self.entropy_coef = entropy_coef
        self.recurrent_module = ours.Recurrence(**network_args)

    @property
    def recurrent_hidden_state_size(self):
        return sum(astuple(self.recurrent_module.state_sizes))

    @property
    def is_recurrent(self):
        return True

    def forward(
        self, inputs, rnn_hxs, masks, deterministic=False, action=None, **kwargs
    ):
        N = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        rm = self.recurrent_module
        hx = rm.parse_hidden(all_hxs)
        actions = [hx.d, hx.dg, hx.a]
        dists = JointDistribution(
            FixedCategorical(probs=hx.d_probs),
            FixedCategorical(probs=hx.dg_probs),
            FixedCategorical(probs=hx.a_probs),
        )

        aux_loss = (
            self.gate_coef * hx.dg_probs[:, 1].mean()
            - self.entropy_coef * dists.entropy()
        )

        rnn_hxs = torch.cat(astuple(hx), dim=-1)
        return AgentOutputs(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=dists.log_probs(hx.d, hx.dg, hx.a),
            aux_loss=aux_loss,
            dist=None,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=(dists.entropy())),
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
        hx = self.recurrent_module.parse_hidden(last_hx)
        return hx.v
