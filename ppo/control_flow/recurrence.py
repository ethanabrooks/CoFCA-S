from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.control_flow.env import Obs
from ppo.distributions import Categorical, FixedCategorical
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a p w v h a_probs p_probs")


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
        num_edges,
        num_encoding_layers,
        debug,
        no_scan,
        no_roll,
    ):
        super().__init__()
        self.no_roll = no_roll
        self.no_scan = no_scan
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.ne = num_edges
        nt = int(self.obs_spaces.lines.nvec[0])
        n_a, n_p = map(int, action_space.nvec)
        self.n_a = n_a
        self.embed_task = nn.Embedding(nt, hidden_size)
        self.embed_action = nn.Embedding(n_a, hidden_size)
        self.task_encoder = nn.GRU(
            2 * hidden_size, hidden_size, bidirectional=True, batch_first=True
        )
        in_size = self.obs_sections.condition + hidden_size
        self.gru = nn.GRUCell(in_size, hidden_size)

        layers = []
        for _ in range(num_layers):
            layers.extend([init_(nn.Linear(hidden_size, hidden_size)), activation])
        self.mlp = nn.Sequential(*layers)
        self.option = init_(nn.Linear(hidden_size, self.ne))

        layers = []
        in_size = (2 if no_scan else 1) * hidden_size
        for _ in range(num_encoding_layers - 1):
            layers.extend([init_(nn.Linear(in_size, hidden_size)), activation])
            in_size = hidden_size
        out_size = (self.ne * n_p if no_scan else self.ne) // 2
        self.mlp2 = nn.Sequential(*layers, init_(nn.Linear(in_size, out_size)))

        self.stuff = init_(nn.Linear(hidden_size, 1))
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, n_a)
        self.attention = Categorical(hidden_size, n_a)
        self.state_sizes = RecurrentState(
            a=1, a_probs=n_a, p=1, p_probs=n_p, w=1, v=1, h=hidden_size
        )
        first = torch.zeros(1, 1, 2 * self.obs_sections.lines, 1, 1)
        first[0, 0, 0] = 1
        self.register_buffer("first", first)

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
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)

        # build memory
        lines = inputs.lines.view(T, N, *self.obs_spaces.lines.shape).long()[0, :, :]
        M = self.embed_task(lines.view(-1)).view(
            *lines.shape, self.hidden_size
        )  # n_batch, n_lines, hidden_size

        rolled = []
        nl = self.obs_sections.lines
        for i in range(nl):
            rolled.append(M if self.no_roll else torch.roll(M, shifts=-i, dims=1))
        rolled = torch.cat(rolled, dim=0)
        if self.no_scan:
            _, H = self.task_encoder(rolled)
            H = H.transpose(0, 1).reshape(nl, N, -1)
            P = self.mlp2(H).view(nl, N, nl * 2, self.ne).softmax(2)
        else:
            first = rolled.view(nl, N, nl, self.hidden_size)[:, :, 0].view(
                -1, 1, self.hidden_size
            )
            X = torch.cat([rolled, first.expand_as(rolled)], dim=-1)
            G, _ = self.task_encoder(X)
            G = G.view(nl, N, nl, 2, self.hidden_size)
            B = self.mlp2(G).sigmoid()
            # B = self.mlp2(G).view(nl, N, nl, 2, self.ne // 2).sigmoid()
            # arange = 0.05 * torch.zeros(16).float()
            # arange[0] = 1
            # B[:, :, :, 0] = arange.view(1, 1, -1, 1)
            # B[:, :, :, 1] = arange.flip(0).view(1, 1, -1, 1)
            f, b = torch.unbind(B, dim=3)
            B = torch.stack([f, b.flip(2)], dim=-2)
            B = F.pad(B, [0, 0, 0, 0, B.size(2), 0])
            # B = B.view(nl, N, 2 * nl, 2, self.ne)
            last = self.first.flip(2)
            zero_last = (1 - last) * B
            B = zero_last + last  # this ensures that the last B is 1
            rolled = torch.roll(zero_last, shifts=1, dims=2)
            C = torch.cumprod(1 - rolled, dim=2)
            P = B * C
            # P = P.view(nl, N, nl, 2, self.ne)
            f, b = torch.unbind(P, dim=3)
            P = torch.cat([b.flip(2), f], dim=3)
            P = P.roll(shifts=(-1, 1), dims=(0, 2))

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        w = hx.w.long().squeeze(-1)
        a = hx.a.long().squeeze(-1)
        a[new_episode] = 0
        R = torch.arange(N, device=rnn_hxs.device)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        W = torch.cat([actions[:, :, 1], hx.p.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("w", w)
            x = [inputs.condition[t], M[R, w]]
            h = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.mlp(h))
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)
            o = self.option(z).softmax(dim=-1)
            self.print("o", torch.round(10 * o))
            g = P[w, R]
            half1 = g.size(1) // 2
            half2 = g.size(2) // 2
            self.print(torch.round(10 * g)[:, :half1, :half2])
            self.print(torch.round(10 * g)[:, half1:, half2:])
            p = (g @ o.unsqueeze(-1)).squeeze(-1)
            p_dist = FixedCategorical(probs=p)
            # p_probs = torch.round(p_dist.probs * 10).flatten()
            self.sample_new(W[t], p_dist)
            w = w + W[t].clone() - nl
            w = torch.clamp(w, min=0, max=nl - 1)
            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                w=w,
                a_probs=a_dist.probs,
                p=W[t],
                p_probs=p_dist.probs,
            )
