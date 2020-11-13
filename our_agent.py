import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

import agents
import ours
from distribution_modules import FixedCategorical
from agents import AgentOutputs, NNBase
from data_types import Command
from distributions import JointDistribution, ConditionalCategorical


class Agent(agents.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        observation_space,
        action_space,
        **network_args,
    ):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        self.recurrent_module = ours.Recurrence(
            observation_space=observation_space,
            action_space=action_space,
            **network_args,
        )

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
        actions = [hx.d, hx.dg, hx.a]
        dists = JointDistribution(
            FixedCategorical(logits=hx.d_probs),
            FixedCategorical(logits=hx.dg_probs),
            ConditionalCategorical(hx.options_probs, hx.choices_probs),
        )

        aux_loss = -self.entropy_coef * dists.entropy()
        # if probs.dg is not None:
        #     aux_loss += rm.gate_coef * hx.dg_probs[:, 1].mean()

        rnn_hxs = torch.cat(hx, dim=-1)
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
