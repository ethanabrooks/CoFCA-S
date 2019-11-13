from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.control_flow.env import Obs
from ppo.distributions import Categorical
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
        debug,
        baseline,
        reduceG,
        w_equals_active,
    ):
        super().__init__()
        if reduceG is None:
            self.w_equals_active = w_equals_active
        else:
            self.w_equals_active = True
        self.reduceG = reduceG
        self.baseline = baseline
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        nl = int(self.obs_spaces.lines.nvec[0])
        na = int(action_space.nvec[0])
        self.embed_task = nn.Embedding(nl, hidden_size)
        self.embed_action = nn.Embedding(na, hidden_size)
        self.task_encoder = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        in_size = self.obs_sections.condition + 2 * hidden_size
        if reduceG is not None:
            in_size += hidden_size
        self.gru = nn.GRUCell(in_size, hidden_size)

        layers = []
        for _ in range(num_layers + 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation])
        self.mlp = nn.Sequential(*layers)

        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, na)
        self.attention = Categorical(hidden_size, na)
        self.state_sizes = RecurrentState(
            a=1, a_probs=na, p=1, p_probs=na, w=1, v=1, h=hidden_size
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
        M = self.embed_task(lines.view(-1)).view(
            *lines.shape, self.hidden_size
        )  # n_batch, n_lines, hidden_size
        gru_input = M.transpose(0, 1)

        G = []
        for i in range(self.obs_sections.lines):
            _, g = self.task_encoder(torch.roll(gru_input, shifts=-i, dims=0))
            G.append(g)
        G = torch.stack(G, dim=0)  # ns, 2, nb, h
        G = G.transpose(1, 2).reshape(G.size(0), N, -1)  # ns, nb, 2*h
        g = None
        if self.reduceG == "first":
            g = G[0]
        elif self.reduceG == "sum":
            g = G.sum(dim=0)
        elif self.reduceG == "mean":
            g = G.mean(dim=0)
        elif self.reduceG == "max":
            g = G.max(dim=0).values
        else:
            assert self.reduceG is None

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        w = hx.w.long().squeeze(-1)
        a = hx.a.long().squeeze(-1)
        p_probs = hx.p_probs
        a[new_episode] = 0
        R = torch.arange(N, device=rnn_hxs.device)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        P = torch.cat([actions[:, :, 1], hx.p.view(1, N)], dim=0).long()
        active = inputs.active.squeeze(-1).long()

        for t in range(T):
            if self.reduceG is None:
                if self.w_equals_active:
                    w = active[t]
                g = G[w, R]
            x = [inputs.condition[t], g]
            if self.reduceG is not None:
                x.append(self.embed_action(A[t - 1].clone()))
            h = self.gru(torch.cat(x, dim=-1), h)
            z = self.mlp(h)
            self.print("active")
            self.print(inputs.active[t])
            a_dist = self.actor(z)
            self.print("probs")
            self.print(torch.round(a_dist.probs * 10))
            self.sample_new(A[t], a_dist)
            if not self.w_equals_active:
                p_dist = self.attention(self.embed_action(A[t].clone()))
                self.sample_new(P[t], p_dist)
                w = w + P[t].clone()
                w = torch.clamp(w, min=0, max=self.obs_sections.lines - 1)
                p_probs = p_dist.probs
            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                w=w,
                a_probs=a_dist.probs,
                p=P[t],
                p_probs=p_probs,
            )
