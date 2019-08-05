import torch

# noinspection PyMissingConstructor
import ppo.agent
from ppo.events.wrapper import Obs
from torch import nn as nn

from ppo.agent import AgentValues, NNBase
from ppo.distributions import Categorical, FixedCategorical
from ppo.events.recurrence import RecurrentState
from ppo.layers import Flatten
from ppo.utils import init_
import torch.nn.functional as F


class DebugAgent(ppo.agent.Agent, NNBase):
    def __init__(self, obs_shape, action_space, entropy_coef, **network_args):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        self.recurrent_module = Recurrence(
            *obs_shape, action_space=action_space, **network_args
        )

    @property
    def recurrent_hidden_state_size(self):
        return 1  # TODO

    @property
    def is_recurrent(self):
        return True  # TODO

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        hx = self.recurrent_module(inputs, rnn_hxs, masks)
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
    def __init__(
        self,
        d,
        h,
        w,
        action_space,
        activation,
        hidden_size,
        num_layers,
        recurrent=False,
    ):
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
        self.action_size = 1
        self.obs_shape = d, h, w

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def parse_inputs(self, inputs: torch.Tensor) -> Obs:
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def forward(self, inputs, rnn_hxs, masks, action=None):
        inputs = inputs.view(1, inputs.size(0), -1)
        if action is None:
            y = F.pad(inputs, [0, self.action_size], "constant", -1)
        else:
            y = torch.cat([inputs, action.float()], dim=-1)
        inputs = y
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        # inputs = inputs._replace(base=inputs.base.view(T, N, *self.obs_shape))
        inputs = inputs.view(N, *self.obs_shape)
        s = self.f(inputs)
        dist = self.actor(s)
        A = actions.long()
        for t in range(T):
            self.sample_new(A[t], dist)
            return RecurrentState(
                a=A[t], a_probs=dist.probs, v=self.critic(s), s=s, p=None
            )

    @property
    def output_size(self):
        return self._hidden_size
