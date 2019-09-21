from collections import namedtuple

import numpy as np
import torch
from gym.spaces import Box
from torch import nn as nn

from ppo.distributions import Categorical
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a probs v")
# "planned_probs plan v t state h model_loss"


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_layers,
        recurrent,
        activation,
    ):
        # recurrent_module = nn.GRU if recurrent else None
        num_inputs = int(np.prod(observation_space.shape))
        super().__init__()
        self.action_size = 1

        self.state_sizes = RecurrentState(a=1, v=1, probs=action_space.n)

        layers = []
        in_size = num_inputs
        for _ in range(num_layers):
            layers += [activation, init_(nn.Linear(in_size, hidden_size))]
            in_size = hidden_size
        self.embed1 = nn.Sequential(*layers)

        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, action_space.n)
        self.continuous = isinstance(action_space, Box)

        self.train()

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, rnn_hxs):
        return self.pack(self.inner_loop(inputs, rnn_hxs=rnn_hxs))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def pack(self, hxs):
        def pack():
            for name, size, hx in zip(
                RecurrentState._fields, self.state_sizes, zip(*hxs)
            ):
                x = torch.stack(hx).float()
                assert np.prod(x.shape[2:]) == size
                yield x.view(*x.shape[:2], -1)

        hx = torch.cat(list(pack()), dim=-1)
        return hx, hx[-1:]

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        rnn_hxs = self.parse_hidden(rnn_hxs)
        for _x in rnn_hxs:
            _x.squeeze_(0)

        A = actions.long()[:, :, 0]
        for t in range(T):
            x = self.embed1(inputs[t])
            dist = self.actor(x)
            self.sample_new(A[t], dist)
            yield RecurrentState(a=A[t], probs=dist.probs, v=self.critic(x))
