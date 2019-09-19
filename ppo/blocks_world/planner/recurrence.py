from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from ppo.distributions import FixedCategorical, Categorical
from ppo.layers import Concat
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a a_probs v h r wr u ww M p L")
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
        embedding_size,
        num_heads,
        planning_steps,
        num_model_layers,
        num_embedding_layers,
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())

        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_space.n,
            v=1,
            h=num_layers * hidden_size,
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
        self.gru = nn.GRU(
            int(embedding_size * np.prod(nvec.shape)) + num_heads * slot_size,
            hidden_size,
            num_layers,
        )
        self.Wxi = nn.Sequential(
            activation,
            init_(nn.Linear(num_layers * hidden_size, sum(self.xi_sections))),
        )
        self.actor = Categorical(num_layers * hidden_size, action_space.n)
        self.critic = init_(nn.Linear(num_layers * hidden_size, 1))

        self.register_buffer("mem_one_hots", torch.eye(num_slots))
        self.embeddings = nn.Embedding(int(nvec.max()), embedding_size)

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
        inputs = self.embeddings(inputs.long())

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
        h = (
            hx.h.view(N, self.gru.num_layers, self.gru.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )

        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long()

        for t in range(T):
            x = torch.cat([inputs[t].view(1, N, -1), r.view(1, N, -1)], dim=-1)
            _, h = self.gru(x, h)
            hT = h.transpose(0, 1).reshape(N, -1)  # switch layer and batch dims
            xi = self.Wxi(hT)
            Kr, br, kw, bw, e, v, free, ga, gw, Pi = xi.squeeze(0).split(
                self.xi_sections, dim=-1
            )
            # preprocess
            br = F.softplus(br.view(N, self.num_heads))
            bw = F.softplus(bw)
            e = e.sigmoid().view(N, 1, self.slot_size)
            v = v.view(N, 1, self.slot_size)
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
            phi_prod = F.pad(phi_prod, [1, 0], value=1)[:, :-1]
            unsorted_phi_prod = phi_prod.scatter(-1, phi.indices, phi_prod)
            a = (1 - u) * unsorted_phi_prod  # page 8 left column
            cw = (
                bw * F.cosine_similarity(M, kw.view(N, 1, self.slot_size), dim=-1)
            ).softmax(dim=-1)
            ww = gw * (ga * a + (1 - ga) * cw)
            ww1 = ww.unsqueeze(-1)
            ww2 = ww.unsqueeze(-2)
            M = M * (1 - ww1 * e) + ww1 * v
            # page 7 right column

            # read
            L = (1 - ww1 - ww2) * L + ww2 * p.unsqueeze(-1)
            L = (1 - self.mem_one_hots).unsqueeze(0) * L  # zero out L[i, i]
            p = (1 - ww.sum(-1, keepdim=True)) * p + ww
            b = wr @ L
            f = wr @ L.transpose(1, 2)
            Kr = Kr.view(N, self.num_heads, 1, self.slot_size)
            cr = (
                br.unsqueeze(-1) * F.cosine_similarity(M.unsqueeze(1), Kr, dim=-1)
            ).softmax(-1)
            wr = Pi[0] * b + Pi[1] * cr + Pi[2] * f
            r = (wr @ M).view(N, -1)

            # act
            x = torch.cat([hT, r], dim=-1)
            dist = self.actor(hT)  # page 7 left column
            value = self.critic(hT)
            self.sample_new(A[t], dist)

            yield RecurrentState(
                a=A[t],
                a_probs=dist.probs,
                v=value,
                h=hT,
                r=r,
                wr=wr,
                u=u,
                ww=ww,
                M=M,
                p=p,
                L=L,
            )
