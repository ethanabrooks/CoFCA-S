from dataclasses import replace

import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

import agents
import ours
from agents import AgentOutputs, NNBase
from ours import RawAction
from distributions import FixedCategorical
from utils import astuple


class Agent(agents.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        observation_space,
        action_space,
        gate_coef,
        **network_args,
    ):
        nn.Module.__init__(self)
        self.gate_coef = gate_coef
        self.entropy_coef = entropy_coef
        self.recurrent_module = ours.Recurrence(
            observation_space=observation_space,
            action_space=action_space,
            **network_args,
        )

    @property
    def recurrent_hidden_state_size(self):
        return sum(astuple(self.recurrent_module.state_sizes))

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
        X = RawAction(a=hx.a, delta=hx.d, dg=hx.dg, ptr=hx.p)
        probs = RawAction(a=hx.a_probs, delta=hx.d_probs, dg=hx.dg_probs, ptr=None)

        dists = [(p if p is None else FixedCategorical(p)) for p in astuple(probs)]
        action_log_probs = sum(
            dist.log_probs(x) for dist, x in zip(dists, astuple(X)) if dist is not None
        )
        entropy = sum([dist.entropy() for dist in dists if dist is not None]).mean()
        aux_loss = -self.entropy_coef * entropy
        if probs.dg is not None:
            aux_loss += self.gate_coef * hx.dg_probs[:, 1].mean()

        rnn_hxs = torch.cat(astuple(hx), dim=-1)
        action = torch.cat(astuple(X), dim=-1)
        return AgentOutputs(
            value=hx.v,
            action=action,
            action_log_probs=action_log_probs,
            aux_loss=aux_loss,
            dist=None,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
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
