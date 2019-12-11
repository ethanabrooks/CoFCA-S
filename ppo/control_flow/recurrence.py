from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box
from torch import nn as nn

from ppo.control_flow.env import Obs
from ppo.distributions import Categorical, FixedCategorical
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a d p v h a_probs p_probs")


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
        eval_lines,
        activation,
        hidden_size,
        num_layers,
        num_edges,
        num_encoding_layers,
        debug,
        no_scan,
        no_roll,
        no_pointer,
        include_action,
    ):
        super().__init__()
        self.include_action = include_action
        self.no_pointer = no_pointer
        self.no_roll = no_roll
        self.no_scan = no_scan or no_pointer  # no scan if no pointer
        self.obs_spaces = Obs(**observation_space.spaces)
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        self._evaluating = False
        self._obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.eval_lines = eval_lines
        self.train_lines = self._obs_sections.lines

        # networks
        self.ne = num_edges
        n_lt = int(self.obs_spaces.lines.nvec[0])
        n_a, n_p = map(int, action_space.nvec)
        self.n_a = n_a
        self.embed_task = nn.Embedding(n_lt, hidden_size)
        self.embed_action = nn.Embedding(n_a, hidden_size)
        self.task_encoder = nn.GRU(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )
        if no_pointer:
            in_size = 3 * hidden_size
        elif include_action:
            in_size = 2 * hidden_size
        else:
            in_size = hidden_size
        in_size += self.obs_sections.obs
        self.gru = nn.GRUCell(in_size, hidden_size)

        layers = []
        for _ in range(num_layers):
            layers.extend([init_(nn.Linear(hidden_size, hidden_size)), activation])
        self.zeta = nn.Sequential(*layers)
        self.upsilon = init_(nn.Linear(hidden_size, self.ne))

        layers = []
        in_size = (2 if self.no_scan else 1) * hidden_size
        for _ in range(num_encoding_layers - 1):
            layers.extend([init_(nn.Linear(in_size, hidden_size)), activation])
            in_size = hidden_size
        out_size = self.ne * 2 * self.train_lines if self.no_scan else self.ne
        self.beta = nn.Sequential(*layers, init_(nn.Linear(in_size, out_size)))

        self.stuff = init_(nn.Linear(hidden_size, 1))
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, n_a)
        self.attention = Categorical(hidden_size, n_a)
        self._state_sizes = RecurrentState(
            a=1, a_probs=n_a, d=1, p_probs=2 * self.train_lines, p=1, v=1, h=hidden_size
        )

    @property
    def state_sizes(self):
        if self.no_scan:
            return self._state_sizes
        return self._state_sizes._replace(p_probs=2 * self.n_lines)

    @property
    def obs_sections(self):
        return self._obs_sections._replace(lines=self.n_lines)

    @property
    def n_lines(self):
        return (1 + self.eval_lines) if self._evaluating else self.train_lines

    @contextmanager
    def evaluating(self):
        self._evaluating = True
        yield self
        self._evaluating = False

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
        T, N, dim = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)

        # build memory
        lines = inputs.lines.view(T, N, self.obs_sections.lines).long()[0, :, :]
        M = self.embed_task(lines.view(-1)).view(
            *lines.shape, self.hidden_size
        )  # n_batch, n_lines, hidden_size

        rolled = []
        nl = self.obs_sections.lines
        for i in range(nl):
            rolled.append(M if self.no_roll else torch.roll(M, shifts=-i, dims=1))
        rolled = torch.cat(rolled, dim=0)
        G, H = self.task_encoder(rolled)
        H = H.transpose(0, 1).reshape(nl, N, -1)
        if self.no_scan:
            P = self.beta(H).view(nl, N, -1, self.ne).softmax(2)
        else:
            G = G.view(nl, N, nl, 2, self.hidden_size)
            B = self.beta(G)
            # arange = 0.05 * torch.zeros(15).float()
            # arange[0] = 1
            # B[:, :, :, 0] = arange.view(1, 1, -1, 1)
            # B[:, :, :, 1] = arange.flip(0).view(1, 1, -1, 1)
            f, b = torch.unbind(B.sigmoid(), dim=3)
            B = torch.stack([f, b.flip(2)], dim=-2)
            B = B.view(nl, N, 2 * nl, self.ne)
            last = torch.zeros_like(B)
            last[:, :, -1] = 1
            zero_last = (1 - last) * B
            B = zero_last + last  # this ensures that the last B is 1
            rolled = torch.roll(zero_last, shifts=1, dims=2)
            C = torch.cumprod(1 - rolled, dim=2)
            P = B * C
            P = P.view(nl, N, nl, 2, self.ne)
            f, b = torch.unbind(P, dim=3)
            P = torch.cat([b.flip(2), f], dim=2)
            P = P.roll(shifts=(-1, 1), dims=(0, 2))

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        p = hx.p.long().squeeze(-1)
        a = hx.a.long().squeeze(-1)
        a[new_episode] = 0
        R = torch.arange(N, device=rnn_hxs.device)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            x = [inputs.obs[t], H.sum(0) if self.no_pointer else M[R, p]]
            if self.no_pointer or self.include_action:
                x += [self.embed_action(A[t - 1].clone())]
            h = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.zeta(h))
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)
            u = self.upsilon(z).softmax(dim=-1)
            self.print("o", torch.round(10 * u))
            g = P[p, R]
            half1 = g.size(1) // 2
            self.print(torch.round(10 * g)[0, half1:])
            self.print(torch.round(10 * g)[0, :half1])
            p_dist = FixedCategorical(probs=((g @ u.unsqueeze(-1)).squeeze(-1)))
            # p_probs = torch.round(p_dist.probs * 10).flatten()
            self.sample_new(D[t], p_dist)
            p = p + D[t].clone() - nl
            p = torch.clamp(p, min=0, max=nl - 1)
            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                p_probs=p_dist.probs,
            )
