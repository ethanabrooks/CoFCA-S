import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import DiagGaussian
from ppo.picture_hanging.env import Obs
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
        bottleneck,
        bidirectional,
    ):
        super().__init__()
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.gru = nn.GRU(1, hidden_size, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.conv = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.bottleneck = init_(nn.Linear(hidden_size * num_directions, bottleneck))
        layers = []
        in_size = self.obs_sections.obs + hidden_size * num_directions
        for i in range(num_layers):
            layers += [init_(nn.Linear(in_size, hidden_size)), activation]
            in_size = hidden_size
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

    def parse_inputs(self, inputs: torch.Tensor):
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

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
        inputs = self.parse_inputs(inputs)
        M, Mn = self.gru(inputs.sizes[0].T.unsqueeze(-1))

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        P = hx.p.squeeze(1).long()
        R = torch.arange(P.size(0), device=P.device)
        A = actions.clone()

        for t in range(T):
            r = M[P, R]
            h = torch.cat([Mn.transpose(0, 1).reshape(N, -1), inputs.obs[t]], dim=-1)
            v = self.critic(h)
            dist = self.actor(h)
            self.sample_new(A[t], dist)
            yield RecurrentState(
                a=A[t],
                loc=dist.loc,
                scale=dist.scale,
                v=v,
                h=hx.h,
                p=(P + 1) % (M.size(0)),
            )
