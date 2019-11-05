import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import AgentValues, NNBase
from ppo.distributions import FixedCategorical

# noinspection PyMissingConstructor
from ppo.control_flow.baselines import oh_et_al
from ppo.control_flow.recurrence import RecurrentState, Recurrence


class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, entropy_coef, recurrent, baseline, **network_args):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        if baseline == "oh-et-al":
            self.recurrent_module = oh_et_al.Recurrence(**network_args)
        else:
            self.recurrent_module = Recurrence(
                **network_args, baseline=baseline == "no-attention"
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
        hx = RecurrentState(*rm.parse_hidden(all_hxs))
        a_dist = FixedCategorical(hx.a_probs)
        p_dist = FixedCategorical(hx.p_probs)
        action_log_probs = a_dist.log_probs(hx.a) + p_dist.log_probs(hx.p)
        entropy = (a_dist.entropy() + p_dist.entropy()).mean()
        return AgentValues(
            value=hx.v,
            action=torch.cat([hx.a, hx.p], dim=-1),
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
        return super()._forward_gru(y, hxs, masks)

    def get_value(self, inputs, rnn_hxs, masks):
        all_hxs, last_hx = self._forward_gru(
            inputs.view(inputs.size(0), -1), rnn_hxs, masks
        )
        return self.recurrent_module.parse_hidden(last_hx).v
