from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from ppo.distributions import FixedCategorical, Categorical
from ppo.layers import Concat
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a a_probs v r wr u ww M p L")
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
        num_slots,
        slot_size,
        num_heads,
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_heads = num_heads

        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_space.n,
            v=1,
            r=num_heads * slot_size,
            wr=num_heads * num_slots,
            u=num_slots,
            ww=num_slots,
            M=num_slots * slot_size,
            p=num_slots,
            L=num_slots * num_slots,
        )
        self.xi_sections = XiSections(
            Kr=num_heads * slot_size,
            Br=num_heads,
            kw=slot_size,
            bw=1,
            e=slot_size,
            v=slot_size,
            F_hat=num_heads,
            ga=1,
            gw=1,
            Pi=3 * num_heads,
        )

        # networks
        assert num_layers > 0
        self.gru = nn.GRU(observation_space.nvec.size, hidden_size, num_layers)
        self.f1 = nn.Sequential(
            init_(nn.Linear(num_heads * slot_size, num_layers * hidden_size)),
            activation,
        )
        self.f2 = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, sum(self.xi_sections)))
        )
        self.actor = Categorical(num_heads * slot_size, action_space.n)
        self.critic = init_(nn.Linear(num_heads * slot_size, 1))
        self.register_buffer("mem_one_hots", torch.eye(num_slots))

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

        wr = hx.wr.view(N, self.num_heads, self.num_slots)
        u = hx.u
        ww = hx.ww.view(N, self.num_slots)
        r = hx.r.view(N, -1).unsqueeze(0)
        M = hx.M.view(N, self.num_slots, self.slot_size)
        p = hx.p
        L = hx.L.view(N, self.num_slots, self.num_slots)

        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            h = self.f1(r.view(N, -1))
            h, _ = self.gru(inputs[t].unsqueeze(0), h.view(self.gru.num_layers, N, -1))
            xi = self.f2(h.view(N, -1))
            Kr, br, kw, bw, e, v, free, ga, gw, Pi = xi.squeeze(0).split(
                self.xi_sections, dim=-1
            )
            br = F.softplus(br.view(N, self.num_heads))
            bw = F.softplus(bw)
            e = e.sigmoid()
            free = free.sigmoid()
            ga = ga.sigmoid()
            gw = gw.sigmoid()
            Pi = (
                Pi.view(N, self.num_heads, -1)
                .permute(2, 0, 1)
                .softmax(dim=0)
                .unsqueeze(-1)
            )

            # write
            psi = (1 - free.unsqueeze(-1) * wr).prod(dim=1)  # page 8 left column
            u = (u + (1 - u) * ww) * psi
            phi = u.sort(dim=-1)
            phi_prod = torch.cumprod(phi.values, dim=-1)
            unsorted_phi_prod = phi_prod.scatter(-1, phi.indices, phi_prod)
            a = (1 - u) * unsorted_phi_prod  # page 8 left column
            cw = (bw * F.cosine_similarity(M, kw.unsqueeze(1), dim=-1)).softmax(dim=-1)
            ww = gw * (ga * a + (1 - ga) * cw)
            ww1 = ww.unsqueeze(-1)
            ww2 = ww.unsqueeze(-2)
            M = M * (1 - ww1 * e.unsqueeze(1)) + ww1 * v.unsqueeze(1)
            # page 7 right column

            # read
            p = (1 - ww.sum(-1, keepdim=True)) * p + ww
            # TODO: what if we took out ww1 (or maybe ww2)?
            L = (1 - ww1 - ww2) * L + ww1 * p.unsqueeze(-1)
            L = (1 - self.mem_one_hots).unsqueeze(0) * L  # zero out L[i, i]
            b = wr @ L
            f = wr @ L.transpose(1, 2)
            Kr = Kr.view(N, self.num_heads, 1, self.slot_size)
            cr = (
                br.unsqueeze(-1) * F.cosine_similarity(M.unsqueeze(1), Kr, dim=-1)
            ).softmax(-1)
            wr = Pi[0] * b + Pi[1] * cr + Pi[2] * f
            r = wr @ M

            # act
            dist = self.actor(r.view(N, -1))
            value = self.critic(r.view(N, -1))
            self.sample_new(A[t], dist)
            yield RecurrentState(
                a=A[t],
                a_probs=dist.probs,
                v=value,
                r=r,
                wr=wr,
                u=u,
                ww=ww,
                M=M,
                p=p,
                L=L,
            )
