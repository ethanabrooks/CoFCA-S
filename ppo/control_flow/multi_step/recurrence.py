from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.control_flow.recurrence
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_
import numpy as np

RecurrentState = namedtuple(
    "RecurrentState", "a d ag dg p v h a_probs d_probs ag_probs dg_probs"
)


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def __init__(self, hidden_size, gate_coef, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.gate_coef = gate_coef
        self.action_size = 4
        d = self.obs_spaces.obs.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(d, hidden_size, kernel_size=3, padding=1),
            nn.MaxPool2d(self.obs_spaces.obs.shape[1:]),
            nn.ReLU(),
        )
        self.d_gate = Categorical(hidden_size, 2)
        self.a_gate = Categorical(hidden_size, 2)
        self._state_sizes = RecurrentState(
            **self._state_sizes._asdict(), ag_probs=2, dg_probs=2, ag=1, dg=1
        )
        ones = torch.ones(1, dtype=torch.long)
        self.register_buffer("ones", ones)

    @property
    def gru_in_size(self):
        in_size = self.hidden_size
        if self.no_pointer:
            in_size += 2 * self.hidden_size
        else:
            in_size += self.encoder_hidden_size
        if self.no_pointer or self.include_action:
            in_size += self.hidden_size
        return in_size

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

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def inner_loop(self, inputs, rnn_hxs):
        T, N, dim = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)
        inputs = inputs._replace(obs=inputs.obs.view(T, N, *self.obs_spaces.obs.shape))

        # build memory
        lines = inputs.lines.view(T, N, self.obs_sections.lines).long()[0, :, :]
        M = self.embed_task(lines.view(-1)).view(
            *lines.shape, self.encoder_hidden_size
        )  # n_batch, n_lines, hidden_size

        rolled = []
        nl = self.obs_sections.lines
        for i in range(nl):
            rolled.append(M if self.no_roll else torch.roll(M, shifts=-i, dims=1))
        rolled = torch.cat(rolled, dim=0)
        G, H = self.task_encoder(rolled)
        H = H.transpose(0, 1).reshape(nl, N, -1)
        last = torch.zeros(nl, N, 2 * nl, self.ne, device=rnn_hxs.device)
        last[:, :, -1] = 1
        if self.no_scan:
            P = self.beta(H).view(nl, N, -1, self.ne).softmax(2)
            half = P.size(2) // 2
        else:
            G = G.view(nl, N, nl, 2, self.encoder_hidden_size)
            B = self.beta(G).sigmoid()
            # arange = torch.zeros(6).float()
            # arange[0] = 1
            # arange[1] = 1
            # B[:, :, :, 0] = 1  # arange.view(1, 1, -1, 1)
            # B[:, :, :, 1] = 1
            f, b = torch.unbind(B, dim=3)
            B = torch.stack([f.roll(shifts=-1, dims=2), b.flip(2)], dim=-2)
            B = B.view(nl, N, 2 * nl, self.ne)
            zero_last = (1 - last) * B
            B = zero_last + last  # this ensures that the last B is 1
            rolled = torch.roll(zero_last, shifts=1, dims=2)
            C = torch.cumprod(1 - rolled, dim=2)
            P = B * C
            P = P.view(nl, N, nl, 2, self.ne)
            f, b = torch.unbind(P, dim=3)
            half = b.size(2)
            P = torch.cat([b.flip(2), f.roll(shifts=1, dims=2)], dim=2)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        p = hx.p.long().squeeze(-1)
        hx.a[new_episode] = self.n_a - 1
        ag_probs = hx.ag_probs
        ag_probs[new_episode, 1] = 1
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()
        AG = torch.cat([actions[:, :, 2], hx.ag.view(1, N)], dim=0).long()
        DG = torch.cat([actions[:, :, 3], hx.dg.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            obs = self.conv(inputs.obs[t]).view(N, -1)
            x = [obs, H.sum(0) if self.no_pointer else M[R, p]]
            if self.no_pointer or self.include_action:
                x += [self.embed_action(A[t - 1].clone())]
            h = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.zeta(h))

            def gate(gate, new, old):
                old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
                return FixedCategorical(probs=gate * new + (1 - gate) * old)

            a_gate = self.a_gate(z)
            self.sample_new(AG[t], a_gate)
            ag = AG[t].unsqueeze(-1).float()
            a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            self.sample_new(A[t], a_dist)
            u = self.upsilon(z).softmax(dim=-1)
            w = P[p, R]
            d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)
            d_gate = self.d_gate(z)
            self.sample_new(DG[t], d_gate)
            dg = DG[t].unsqueeze(-1).float()
            d_dist = gate(dg, d_probs, ones * half)
            self.sample_new(D[t], d_dist)
            p = p + D[t].clone() - half
            if self.clamp_p:
                p = torch.clamp(p, min=0, max=nl - 1)
            else:
                p = p % nl

            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=d_dist.probs,
                ag_probs=a_gate.probs,
                dg_probs=d_gate.probs,
                ag=ag,
                dg=dg,
            )
