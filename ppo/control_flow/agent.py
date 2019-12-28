import torch
import torch.jit
from gym.spaces import Box
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import AgentValues, NNBase

# noinspection PyMissingConstructor
from ppo.control_flow.baselines import oh_et_al
from ppo.control_flow.recurrence import RecurrentState
import ppo.control_flow.recurrence
import ppo.control_flow.multi_step.recurrence
import ppo.control_flow.simple
from ppo.distributions import FixedCategorical


class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        recurrent,
        observation_space,
        include_action,
        use_conv,
        gate_coef,
        no_op_coef,
        **network_args
    ):
        nn.Module.__init__(self)
        self.no_op_coef = no_op_coef
        self.entropy_coef = entropy_coef
        self.multi_step = type(observation_space.spaces["obs"]) is Box
        self.recurrent_module = (
            ppo.control_flow.multi_step.recurrence.Recurrence(
                include_action=True,
                observation_space=observation_space,
                use_conv=use_conv,
                gate_coef=gate_coef,
                **network_args
            )
            if self.multi_step
            else ppo.control_flow.recurrence.Recurrence(
                include_action=include_action,
                observation_space=observation_space,
                **network_args
            )
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
        if rm.no_pointer:
            action_log_probs = a_dist.log_probs(hx.a)
            entropy = a_dist.entropy().mean()
            action = F.pad(hx.a, [0, 1])
        else:
            probs = [hx.a_probs, hx.d_probs, hx.ag_probs, hx.dg_probs]
            X = [hx.a, hx.d, hx.ag, hx.dg]
            dists = [FixedCategorical(p) for p in probs]
            action_log_probs = sum(dist.log_probs(x) for dist, x in zip(dists, X))
            entropy = sum([dist.entropy() for dist in dists]).mean()
            # action_log_probs = a_dist.log_probs(hx.a) + d_dist.log_probs(hx.d)
            # entropy = (a_dist.entropy() + d_dist.entropy()).mean()
            action = torch.cat(X, dim=-1)
        aux_loss = -self.entropy_coef * entropy
        if self.multi_step:
            assert rm.gate_coef is not None
            assert self.no_op_coef is not None
            aux_loss += (
                rm.gate_coef * (hx.ag_probs + hx.dg_probs)[:, 1].mean()
                + self.no_op_coef * hx.a_probs[:, -1].mean()
            )
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
