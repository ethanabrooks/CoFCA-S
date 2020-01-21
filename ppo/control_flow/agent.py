import torch
import torch.jit
from gym.spaces import Box
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import AgentValues, NNBase

from ppo.control_flow.recurrence import RecurrentState
import ppo.control_flow.recurrence
import ppo.control_flow.recurrence
import ppo.control_flow.multi_step.no_pointer
import ppo.control_flow.multi_step.oh_et_al
import ppo.control_flow.simple
from ppo.distributions import FixedCategorical


class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        recurrent,
        observation_space,
        include_action,
        gate_coef,
        no_op_coef,
        baseline,
        **network_args
    ):
        nn.Module.__init__(self)
        self.no_op_coef = no_op_coef
        self.entropy_coef = entropy_coef
        multi_step = type(observation_space.spaces["obs"]) is Box
        if baseline == "no-pointer":
            self.recurrent_module = ppo.control_flow.multi_step.no_pointer.Recurrence(
                include_action=True,
                observation_space=observation_space,
                gate_coef=gate_coef,
                no_pointer=True,
                **network_args
            )
        elif baseline == "oh-et-al":
            self.recurrent_module = ppo.control_flow.multi_step.oh_et_al.Recurrence(
                include_action=True,
                observation_space=observation_space,
                gate_coef=gate_coef,
                no_pointer=True,
                **network_args
            )
        else:
            assert baseline is None
            self.recurrent_module = ppo.control_flow.recurrence.Recurrence(
                include_action=True,
                observation_space=observation_space,
                gate_coef=gate_coef,
                no_pointer=False,
                multi_step=multi_step,
                **network_args
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
        a_dist = FixedCategorical(hx.a_probs)
        probs = [hx.a_probs, hx.d_probs, hx.ag_probs, hx.dg_probs]
        X = [hx.a, hx.d, hx.ag, hx.dg]
        dists = [FixedCategorical(p) for p in probs]
        if type(rm) in (
            ppo.control_flow.multi_step.oh_et_al.Recurrence,
            ppo.control_flow.multi_step.no_pointer.Recurrence,
        ):
            action_log_probs = a_dist.log_probs(hx.a)
            entropy = a_dist.entropy().mean()
            if type(rm) is ppo.control_flow.multi_step.oh_et_al:
                assert rm.gate_coef is not None
                aux_loss = rm.gate_coef * (hx.ag + hx.dg)
            else:
                aux_loss = 0
        else:
            action_log_probs = sum(dist.log_probs(x) for dist, x in zip(dists, X))
            entropy = sum([dist.entropy() for dist in dists]).mean()
            # action_log_probs = a_dist.log_probs(hx.a) + d_dist.log_probs(hx.d)
            # entropy = (a_dist.entropy() + d_dist.entropy()).mean()
            assert rm.gate_coef is not None
            assert self.no_op_coef is not None
            aux_loss = (
                rm.gate_coef * (hx.ag_probs + hx.dg_probs)[:, 1].mean()
                + self.no_op_coef * hx.a_probs[:, -1].mean()
            )
        action = torch.cat(X, dim=-1)
        aux_loss -= self.entropy_coef * entropy
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
