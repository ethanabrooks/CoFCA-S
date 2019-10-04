import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import DiagGaussian
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a loc scale v h p")


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
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.gru = nn.GRU(
            observation_space.shape[0],
            hidden_size,
            # num_layers,
        )
        self.critic = nn.Sequential()
        self.actor = nn.Sequential()
        layers = []
        for i in range(num_layers):
            layers += [init_(nn.Linear(hidden_size, hidden_size)), activation]
        self.actor = nn.Sequential(*layers)
        self.critic = copy.deepcopy(self.actor)
        self.actor.add_module("dist", DiagGaussian(hidden_size, action_space.shape[0]))
        self.critic.add_module("out", init_(nn.Linear(hidden_size, 1)))
        self.state_sizes = RecurrentState(a=1, loc=1, scale=1, p=1, v=1, h=hidden_size)

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

    # def parse_inputs(self, inputs: torch.Tensor):
    #     return ppo.oh_et_al.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            torch.set_printoptions(precision=2, sci_mode=False)
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = (
            hx.h.view(N, self.gru.num_layers, self.gru.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        A = actions.clone()

        for t in range(T):
            hn, h = self.gru(inputs[t].unsqueeze(0), h)
            v = self.critic(hn.squeeze(0))
            a_dist = self.actor(hn.squeeze(0))
            self.sample_new(A[t], a_dist)
            yield RecurrentState(
                a=A[t],
                loc=a_dist.loc,
                scale=a_dist.scale,
                v=v,
                h=h.transpose(0, 1),
                p=hx.p,
            )
