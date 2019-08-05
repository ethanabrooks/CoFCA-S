from torch import nn as nn

from ppo.agent import AgentValues, NNBase
from ppo.distributions import Categorical, FixedCategorical
from ppo.events.recurrence import RecurrentState
from ppo.layers import Flatten
from ppo.utils import init_


# noinspection PyMissingConstructor
import ppo.agent


class DebugAgent(ppo.agent.Agent, NNBase):
    def __init__(self, obs_shape, action_space, entropy_coef, **network_args):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        self.recurrent_module = Recurrence(*obs_shape, **network_args)

    @property
    def recurrent_hidden_state_size(self):
        return 1  # TODO

    @property
    def is_recurrent(self):
        return True  # TODO

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        hx = self.recurrent_module(inputs, rnn_hxs, masks, action=action)
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
        return self.recurrent_module(inputs, rnn_hxs, masks).v


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
    def output_size(self):
        return self._hidden_size
