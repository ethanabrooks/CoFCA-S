from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import FixedCategorical, Categorical
from ppo.control_flow.env import Obs
from ppo.layers import Concat
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a p v h a_probs p_probs")


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
        baseline,
    ):
        super().__init__()
        self.baseline = baseline
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        nl = int(self.obs_spaces.lines.nvec[0])
        self.embeddings = nn.Embedding(nl, hidden_size)
        self.task_encoder = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        # f
        layers = []
        in_size = self.obs_sections.condition + hidden_size
        for _ in range(num_layers + 1):
            layers.extend([init_(nn.Linear(in_size, hidden_size)), activation])
            in_size = hidden_size
        self.f = nn.Sequential(*layers)

        self.na = na = int(action_space.nvec[0])
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.action_embedding = nn.Embedding(na, hidden_size)
        self.pointer = Categorical(hidden_size, self.obs_sections.lines)
        self.actor = Categorical(hidden_size, na)
        self.query = init_(nn.Linear(hidden_size, hidden_size))
        self.state_sizes = RecurrentState(
            a=1, a_probs=na, p=1, p_probs=self.obs_sections.lines, v=1, h=hidden_size
        )

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
        M = self.embeddings(lines.view(-1)).view(
            *lines.shape, self.hidden_size
        )  # n_batch, n_lines, hidden_size
        gru_input = M.transpose(0, 1)

        K = []
        for i in range(self.obs_sections.lines):
            k, _ = self.task_encoder(torch.roll(gru_input, shifts=i, dims=0))
            K.append(k)
        K1 = torch.stack(K, dim=0)  # ns, ns, nb, 2*h
        K2 = K1.view(K1.size(0), K1.size(1), N, 2, -1)  # ns, ns, nb, 2, h
        K3 = K2.permute(2, 0, 1, 3, 4)  # nb, ns, ns, 2, h
        K = K3.reshape(N, K3.size(1), -1, K3.size(-1))

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        a = hx.a.long().squeeze(-1)
        a[new_episode] = 0
        R = torch.arange(N, device=rnn_hxs.device)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        P = torch.cat([actions[:, :, 1], hx.p.view(1, N)], dim=0).long()

        for t in range(T):
            obs = torch.cat(
                [inputs.condition[t], self.action_embedding(A[t - 1].clone())], dim=-1
            )
            h = self.gru(self.f(obs), h)
            p_dist = self.pointer(h)
            self.sample_new(P[t], p_dist)
            q = self.query(h)
            k = K[R, P[t].clone()]
            l = torch.sum(k * q.unsqueeze(1), dim=-1)
            z = torch.sum(l.unsqueeze(-1) * k, dim=1)
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)
            yield RecurrentState(
                a=A[t],
                v=self.critic(h),
                h=h,
                a_probs=a_dist.probs,
                p=P[t],
                p_probs=p_dist.probs,
            )
