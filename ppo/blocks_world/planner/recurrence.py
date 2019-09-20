from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import Categorical, FixedCategorical
from ppo.layers import Flatten
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a probs v t ")


def batch_conv1d(inputs, weights):
    outputs = []
    # one convolution per instance
    n = inputs.shape[0]
    for i in range(n):
        x = inputs[i]
        w = weights[i]
        convolved = F.conv1d(x.reshape(1, 1, -1), w.reshape(1, 1, -1), padding=2)
        outputs.append(convolved.squeeze(0))
    padded = torch.cat(outputs)
    padded[:, 1] = padded[:, 1] + padded[:, 0]
    padded[:, -2] = padded[:, -2] + padded[:, -1]
    return padded[:, 1:-1]


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        activation,
        hidden_size,
        num_layers,
        debug,
        num_slots,
        slot_size,
        embedding_size,
        num_heads,
        planning_steps,
        num_model_layers,
        num_embedding_layers,
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())
        self.num_options = nvec.max()
        self.hidden_size = hidden_size

        self.state_sizes = RecurrentState(a=1, v=1, t=1, probs=action_space.n)

        # networks
        assert num_layers > 0
        self.embed_action = nn.Embedding(int(action_space.n), int(action_space.n))

        layers = [nn.Embedding(nvec.max(), nvec.max()), Flatten()]
        in_size = int(nvec.max() * np.prod(nvec.shape))
        for _ in range(num_embedding_layers):
            layers += [activation, init_(nn.Linear(in_size, hidden_size))]
            in_size = hidden_size
        self.embed1 = nn.Sequential(*layers)
        self.embed2 = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, embedding_size))
        )

        self.actor = Categorical(embedding_size, action_space.n)
        self.critic = init_(nn.Linear(embedding_size, 1))

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, hx):
        return self.pack(self.inner_loop(inputs, rnn_hxs=hx))

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

    def parse_inputs(self, inputs: torch.Tensor):
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, t, *args, **kwargs):
        if self.debug:
            if type(t) == torch.Tensor:
                t = (t * 10.0).round() / 10.0
            print(t, *args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        inputs = inputs.long()

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        A = actions.long()[:, :, 0]

        for t in range(T):
            x = self.embed2(self.embed1(inputs[t]))

            # act
            dist = self.actor(x)
            value = self.critic(x)
            self.sample_new(A[t], dist)
            yield RecurrentState(a=A[t], probs=dist.probs, v=value, t=hx.t + 1)
