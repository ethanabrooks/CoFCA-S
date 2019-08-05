import torch
from gym.spaces import Box, Discrete
from torch import nn as nn

from ppo.agent import AgentValues
from ppo.distributions import Categorical, FixedCategorical
from ppo.distributions import DiagGaussian
from ppo.events.recurrence import RecurrentState
from ppo.layers import Flatten
from ppo.utils import init_


# noinspection PyMissingConstructor


class DebugAgent(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_space,
        recurrent,
        hidden_size,
        entropy_coef,
        **network_args,
    ):
        super().__init__()
        self.entropy_coef = entropy_coef
        self.base = Recurrence(
            *obs_shape, recurrent=recurrent, hidden_size=hidden_size, **network_args
        )

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.continuous = isinstance(action_space, Box)

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        hx = self.base(inputs, rnn_hxs, masks, action=action)
        dist = FixedCategorical(hx.a_probs)
        action_log_probs = dist.log_probs(hx.a)
        entropy = dist.entropy().mean()
        return AgentValues(
            value=hx.v,
            action=hx.a,
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=dist,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    def get_value(self, inputs, rnn_hxs, masks):
        return self.base(inputs, rnn_hxs, masks).v


class Recurrence(nn.Module):
    def __init__(self, d, h, w, activation, hidden_size, num_layers, recurrent=False):
        super().__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.f = nn.Sequential(
            init_(nn.Conv2d(d, hidden_size, kernel_size=1)),
            activation,
            *[
                nn.Sequential(
                    init_(
                        nn.Conv2d(hidden_size, hidden_size, kernel_size=1), activation
                    ),
                    activation,
                )
                for _ in range(num_layers)
            ],
            activation,
            Flatten(),
            init_(nn.Linear(hidden_size * h * w, hidden_size)),
            activation,
        )
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(self.output_size, 5)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, action=None):
        s = self.f(inputs)
        dist = self.actor(s)

        if action is None:
            action = dist.sample()

        return RecurrentState(
            a=action, a_probs=dist.probs, v=self.critic(s), s=s, p=None
        )

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1  # TODO

    @property
    def is_recurrent(self):
        return False  # TODO

    @property
    def output_size(self):
        return self._hidden_size
