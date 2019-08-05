import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

from ppo.agent import AgentValues, NNBase
import ppo.agent
from ppo.distributions import FixedCategorical, Categorical

# noinspection PyMissingConstructor
from ppo.events.recurrence import Recurrence, RecurrentState
from ppo.layers import Flatten
from ppo.utils import init_


class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        entropy_coef,
        hidden_size,
        obs_spaces,
        activation,
        num_layers,
        action_size,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        # self.recurrent_module = self.build_recurrent_module(
        #     **kwargs,
        #     obs_spaces=obs_spaces,
        #     hidden_size=hidden_size,
        #     activation=activation,
        #     num_layers=num_layers,
        # )
        d, h, w = obs_spaces.shape
        self.f = nn.Sequential(
            init_(nn.Conv2d(d, hidden_size, kernel_size=1), activation),
            activation,
            nn.Sequential(
                *[
                    m
                    for _ in range(num_layers)
                    for m in (
                        init_(
                            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
                            activation,
                        ),
                        activation,
                    )
                ]
            ),
            activation,
            Flatten(),
            nn.Linear(hidden_size * h * w, hidden_size),
        )
        self.actor = Categorical(hidden_size, action_size)
        self.critic = init_(nn.Linear(hidden_size, 1))

    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)

        # TODO {{{
        # rm = self.recurrent_module
        # all_hxs, last_hx = self._forward_gru(
        #     inputs.view(N, -1), rnn_hxs, masks, action=action
        # )
        s = self.f(inputs)
        dist = self.actor(s)
        if action is None:
            action = dist.sample()
        hx = RecurrentState(a=action, a_probs=dist.probs, v=self.critic(s), s=s, p=None)
        last_hx = rnn_hxs
        # hx = RecurrentState(*rm.parse_hidden(all_hxs))
        # TODO }}}

        dist = FixedCategorical(hx.a_probs)
        entropy = dist.entropy().mean()
        return AgentValues(
            value=hx.v,
            action=hx.a,
            action_log_probs=dist.log_probs(hx.a).mean(),
            aux_loss=self.entropy_coef * entropy,
            dist=dist,
            rnn_hxs=last_hx,
            log=dict(entropy=entropy),
        )

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        s = self.f(inputs)
        return self.critic(s)
        # all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        # return self.recurrent_module.parse_hidden(last_hx).v

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            y = torch.cat([x, action.float()], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return 0  # TODO sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True
