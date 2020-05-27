import torch
import torch.jit
from gym.spaces import Box
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
import ppo.control_flow.multi_step.abstract_recurrence
import ppo.control_flow.multi_step.ours
import ppo.control_flow.no_pointer
import ppo.control_flow.recurrence
from ppo.agent import AgentValues, NNBase
from ppo.control_flow.env import Action
from ppo.distributions import FixedCategorical


class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        observation_space,
        no_op_coef,
        action_space,
        lower_level,
        **network_args,
    ):
        nn.Module.__init__(self)
        self.lower_level_type = lower_level
        self.no_op_coef = no_op_coef
        self.entropy_coef = entropy_coef
        self.multi_step = type(observation_space.spaces["obs"]) is Box
        if not self.multi_step:
            del network_args["conv_hidden_size"]
            del network_args["gate_coef"]
        elif self.multi_step:
            self.recurrent_module = ppo.control_flow.multi_step.ours.Recurrence(
                observation_space=observation_space,
                action_space=action_space,
                **network_args,
            )
        else:
            self.recurrent_module = ppo.control_flow.recurrence.Recurrence(
                observation_space=observation_space,
                action_space=action_space,
                **network_args,
            )

    @property
    def recurrent_hidden_state_size(self):
        state_sizes = self.recurrent_module.state_sizes
        state_sizes = state_sizes._replace(P=0)
        return sum(state_sizes)

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
        if t is ppo.control_flow.recurrence.Recurrence:
            X = [hx.a, hx.d, hx.p]
            probs = [hx.a_probs, hx.d_probs]
        elif t is ppo.control_flow.multi_step.ours.Recurrence:
            X = Action(upper=hx.a, lower=hx.l, delta=hx.d, dg=hx.dg, ptr=hx.p)
            ll_type = self.lower_level_type
            if ll_type == "train-alone":
                probs = Action(
                    upper=None, lower=hx.l_probs, delta=None, dg=None, ptr=None
                )
            elif ll_type == "train-with-upper":
                probs = Action(
                    upper=hx.a_probs,
                    lower=hx.l_probs,
                    delta=hx.d_probs,
                    dg=hx.dg_probs,
                    ptr=None,
                )
            elif ll_type in ["pre-trained", "hardcoded"]:
                probs = Action(
                    upper=hx.a_probs,
                    lower=None,
                    delta=None if rm.no_pointer else hx.d_probs,
                    dg=hx.dg_probs,
                    ptr=None,
                )
            else:
                raise RuntimeError
        else:
            raise RuntimeError

        dists = [(p if p is None else FixedCategorical(p)) for p in probs]
        action_log_probs = Action(
            *[
                dist.log_probs(x) if dist is not None else None
                for dist, x in zip(dists, X)
            ]
        )
        entropy = sum([dist.entropy() for dist in dists if dist is not None]).mean()
        aux_loss = -self.entropy_coef * entropy
        if probs.upper is not None:
            aux_loss += self.no_op_coef * hx.a_probs[:, -1].mean()
        if probs.dg is not None:
            aux_loss += rm.gate_coef * hx.dg_probs[:, 1].mean()

        P = hx.P.reshape(N, *rm.P_shape())
        rnn_hxs = torch.cat(
            hx._replace(P=torch.tensor([], device=hx.P.device), l=X.lower), dim=-1
        )
        action = torch.cat(X, dim=-1)
        return AgentValues(
            value=hx.va,
            value2=hx.vd,
            value3=hx.vdg,
            action=action,
            action_log_probs=action_log_probs.upper,
            action_log_probs2=action_log_probs.delta,
            action_log_probs3=action_log_probs.dg,
            aux_loss=aux_loss,
            dist=None,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy, P=P),
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
        return hx.va, hx.vd, hx.vdg
