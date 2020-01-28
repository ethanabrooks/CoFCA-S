import torch
import torch.jit
from gym.spaces import Box
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import AgentValues, NNBase

from ppo.control_flow.recurrence import RecurrentState
import ppo.control_flow.recurrence
import ppo.control_flow.gridworld.abstract_recurrence
import ppo.control_flow.gridworld.no_pointer
import ppo.control_flow.gridworld.oh_et_al
import ppo.control_flow.gridworld.ours
import ppo.control_flow.no_pointer
import ppo.control_flow.oh_et_al
from ppo.distributions import FixedCategorical


class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        recurrent,
        observation_space,
        no_op_coef,
        baseline,
        **network_args
    ):
        nn.Module.__init__(self)
        self.no_op_coef = no_op_coef
        self.entropy_coef = entropy_coef
        self.multi_step = type(observation_space.spaces["obs"]) is Box
        if not self.multi_step:
            del network_args["conv_hidden_size"]
            del network_args["use_conv"]
            del network_args["gate_coef"]
        if baseline == "no-pointer":
            del network_args["gate_coef"]
            self.recurrent_module = (
                ppo.control_flow.gridworld.no_pointer.Recurrence
                if self.multi_step
                else ppo.control_flow.no_pointer.Recurrence
            )(observation_space=observation_space, **network_args)
        elif baseline == "oh-et-al":
            self.recurrent_module = (
                ppo.control_flow.gridworld.oh_et_al.Recurrence
                if self.multi_step
                else ppo.control_flow.oh_et_al.Recurrence
            )(observation_space=observation_space, **network_args)
        elif self.multi_step:
            assert baseline is None
            self.recurrent_module = ppo.control_flow.gridworld.ours.Recurrence(
                observation_space=observation_space, **network_args
            )
        else:
            self.recurrent_module = ppo.control_flow.recurrence.Recurrence(
                observation_space=observation_space, **network_args
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
        t = type(rm)
        pad = hx.a
        if t in (
            ppo.control_flow.oh_et_al.Recurrence,
            ppo.control_flow.no_pointer.Recurrence,
        ):
            X = [hx.a, pad, hx.p]
            probs = [hx.a_probs]
        elif t is ppo.control_flow.recurrence.Recurrence:
            X = [hx.a, hx.d, hx.p]
            probs = [hx.a_probs, hx.d_probs]
        elif t is ppo.control_flow.gridworld.no_pointer.Recurrence:
            X = [hx.a, pad, pad, pad, pad]
            probs = [hx.a_probs]
        elif t is ppo.control_flow.gridworld.oh_et_al.Recurrence:
            X = [hx.a, pad, pad, pad, hx.p]
            probs = [hx.a_probs]
        elif t is ppo.control_flow.gridworld.ours.Recurrence:
            X = [hx.a, hx.d, hx.ag, hx.dg, hx.p]
            probs = [hx.a_probs, hx.d_probs, hx.ag_probs, hx.dg_probs]
        else:
            raise RuntimeError
        dists = [FixedCategorical(p) for p in probs]
        action_log_probs = sum(dist.log_probs(x) for dist, x in zip(dists, X))
        entropy = sum([dist.entropy() for dist in dists]).mean()
        aux_loss = (
            self.no_op_coef * hx.a_probs[:, -1].mean() - self.entropy_coef * entropy
        )
        try:
            aux_loss += rm.gate_coef * (hx.ag_probs + hx.dg_probs)[:, 1].mean()
        except AttributeError:
            pass
        action = torch.cat(X, dim=-1)
        return AgentValues(
            value=hx.v,
            action=action,
            action_log_probs=action_log_probs,
            aux_loss=aux_loss,
            dist=None,
            rnn_hxs=last_hx,
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
        return self.recurrent_module.parse_hidden(last_hx).v
