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
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True  # TODO

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        rm = self.recurrent_module
        hx = RecurrentState(*rm.parse_hidden(last_hx))
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

    def _forward_gru(self, x, hxs, masks, action=None):
        x = x.view(1, x.size(0), -1)
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            action.unsqueeze_(0)
            y = torch.cat([x, action.float()], dim=-1)
        return self.recurrent_module(y, hxs)

    def get_value(self, inputs, rnn_hxs, masks):
        all_hxs, last_hx = self._forward_gru(
            inputs.view(inputs.size(0), -1), rnn_hxs, masks
        )
        return self.recurrent_module.parse_hidden(last_hx).v


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
        self.state_sizes = RecurrentState(
            a=1, a_probs=action_space.n, v=1, s=hidden_size, p=1  # TODO
        )

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, hx):
        return self.pack(self.inner_loop(inputs, rnn_hxs=hx))

    @staticmethod
    def pack(hxs):
        def pack():
            for name, hx in RecurrentState(*zip(*hxs))._asdict().items():
                x = torch.stack(hx).float()
                yield x.view(*x.shape[:2], -1)

        l = list(pack())
        hx = torch.cat(l, dim=-1)
        return hx, hx[-1]  # TODO?

    def parse_inputs(self, inputs: torch.Tensor) -> Obs:
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        hx = self.parse_hidden(rnn_hxs)
        for x in hx:
            x.squeeze_(0)
        # inputs = inputs._replace(base=inputs.base.view(T, N, *self.obs_shape))
        inputs = inputs.view(T, N, *self.obs_shape)
        for t in range(T):
            s = self.f(inputs[t])
            dist = self.actor(s)
            A = actions.long()
            self.sample_new(A[t], dist)
            yield RecurrentState(
                a=A[t],
                a_probs=dist.probs,
                v=self.critic(s),
                s=s,
                p=A[t],  # TODO dummy value
            )

    @property
    def output_size(self):
        return self._hidden_size
