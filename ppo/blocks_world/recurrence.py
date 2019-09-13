from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from ppo.distributions import FixedCategorical, Categorical
from ppo.layers import Concat
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a a_probs v h wr u ww M p L")
XiSections = namedtuple("XiSections", "Kr Br kw bw e v F_hat ga gw Pi")


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
        mem_size,
        num_heads,
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.num_heads = num_heads

        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_space.n,
            v=1,
            h=hidden_size,
            wr=num_heads * mem_size,
            u=mem_size,
            ww=mem_size,
            M=mem_size * hidden_size,
            p=mem_size,
            L=mem_size * mem_size,
        )
        self.xi_sections = XiSections(
            Kr=num_heads * hidden_size,
            Br=num_heads,
            kw=hidden_size,
            bw=1,
            e=hidden_size,
            v=hidden_size,
            F_hat=num_heads,
            ga=1,
            gw=1,
            Pi=3 * num_heads,
        )

        # networks
        assert num_layers > 0
        self.gru = nn.GRU(observation_space.nvec.size, hidden_size, num_layers)
        self.f = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, sum(self.xi_sections)))
        )
        self.actor = Categorical(hidden_size, action_space.n)
        self.critic = init_(nn.Linear(hidden_size, 1))

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

        # new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        wr = hx.wr.view(N, self.num_heads, self.mem_size)
        u = hx.u
        ww = hx.ww
        h = hx.h
        M = hx.M.view(N, self.mem_size, self.hidden_size)
        p = hx.p
        L = hx.L

        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            h, _ = self.gru(inputs[t].unsqueeze(0), h.unsqueeze(0))
            xi = self.f(h)
            Kr, Br, kw, bw, e, v, F_hat, ga, gw, Pi = torch.split(
                xi.squeeze(0), self.xi_sections, dim=-1
            )
            Br = F.softplus(Br.view(N, self.num_heads))
            bw = F.softplus(bw)
            e = torch.sigmoid(e)
            F_hat = torch.sigmoid(F_hat)
            ga = torch.sigmoid(ga)
            gw = torch.sigmoid(gw)
            Pi = torch.softmax(Pi.view(N, self.num_heads, -1), dim=-1)

            # write
            unsqueeze_wr = F_hat.unsqueeze(-1) * wr
            import ipdb

            ipdb.set_trace()
            psi = torch.prod(1 - unsqueeze_wr, dim=-1)  # page 8 left column
            import ipdb

            ipdb.set_trace()
            u = (u + ww - u * ww) * psi
            phi = torch.argsort(u, dim=-1)
            a = (1 - u[phi]) * torch.prod(u[phi])  # page 8 left column
            cw = torch.softmax(bw * F.cosine_similarity(M, kw), dim=-1)
            ww = gw * (ga * a + (1 - ga) * cw)
            M = M * (1 - ww @ e) + ww @ v  # page 7 right column

            # read
            p = (1 - ww.sum()) * p + ww
            ww1 = ww.unsqueeze(-1)
            ww2 = ww.unsqueeze(-2)
            L = (1 - ww1 - ww2) * L + ww1 * p.unsqueeze(-2)
            f = L @ wr
            b = L @ wr
            cr = torch.softmax(Br * F.cosine_similarity(M, Kr), dim=-1)
            wr = Pi[1] * b + Pi[2] * cr + Pi[3] * f
            r = M @ wr

            # act
            dist = self.actor(r)
            self.sample_new(A[t], dist)
            yield RecurrentState(
                a=A[t],
                a_probs=dist.probs,
                v=self.critic(h),
                h=h,
                wr=wr,
                u=u,
                ww=ww,
                M=M,
                p=p,
                L=L,
            )
