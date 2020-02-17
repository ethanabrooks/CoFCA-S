from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.nn as nn

import ppo.control_flow.multi_step.abstract_recurrence as abstract_recurrence
import ppo.control_flow.recurrence as recurrence
from ppo.distributions import FixedCategorical, Categorical
import numpy as np

from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState", "a d ag dg p v h h2 a_probs d_probs ag_probs dg_probs"
)


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


class Recurrence(abstract_recurrence.Recurrence, recurrence.Recurrence):
    def __init__(
        self, hidden_size, conv_hidden_size, gate_coef, num_conv_layers, **kwargs
    ):
        self.gate_coef = gate_coef
        self.conv_hidden_size = conv_hidden_size
        recurrence.Recurrence.__init__(self, hidden_size=hidden_size, **kwargs)
        abstract_recurrence.Recurrence.__init__(
            self, conv_hidden_size=conv_hidden_size, num_conv_layers=num_conv_layers,
        )
        self.linear = init_(nn.Linear(conv_hidden_size, self.enc))
        self.d_gate = Categorical(hidden_size, 2)
        self.a_gate = Categorical(hidden_size, 2)
        state_sizes = self.state_sizes._asdict()
        del state_sizes["u"]
        self.state_sizes = RecurrentState(
            **state_sizes, h2=hidden_size, ag_probs=2, dg_probs=2, ag=1, dg=1
        )

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
        nl = len(self.obs_spaces.lines.nvec)
        M = self.build_memory(N, T, inputs)

        P = self.build_P(M, N, rnn_hxs.device, nl)
        half = P.size(2) // 2 if self.no_scan else nl
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        h2 = hx.h2
        p = hx.p.long().squeeze(-1)
        hx.a[new_episode] = self.n_a - 1
        ag_probs = hx.ag_probs
        ag_probs[new_episode, 1] = 1
        R = torch.arange(N, device=(rnn_hxs.device))
        ones = self.ones.expand_as(R)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()
        AG = torch.cat([actions[:, :, 2], hx.ag.view(1, N)], dim=0).long()
        DG = torch.cat([actions[:, :, 3], hx.dg.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            obs = self.preprocess_obs(inputs.obs[t])
            obs = obs * M[R, p]
            x = [obs, self.embed_action(A[t - 1].clone())]
            h = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.zeta(h))
            d_gate = self.d_gate(z)
            self.sample_new(DG[t], d_gate)
            a_gate = self.a_gate(z)
            self.sample_new(AG[t], a_gate)

            h2_ = self.gru(torch.cat(x, dim=-1), h2)
            z = F.relu(self.zeta(h2_))
            u = self.upsilon(z).softmax(dim=-1)
            self.print("u", u)
            w = P[p, R]
            d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)
            dg = DG[t].unsqueeze(-1).float()
            self.print("dg prob", d_gate.probs[:, 1])
            self.print("dg", dg)
            d_dist = gate(dg, d_probs, ones * half)
            self.print("d_probs", d_probs[:, half:])
            self.sample_new(D[t], d_dist)
            p = p + D[t].clone() - half
            p = torch.clamp(p, min=0, max=M.size(1) - 1)

            ag = AG[t].unsqueeze(-1).float()
            a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            self.sample_new(A[t], a_dist)
            self.print("ag prob", a_gate.probs[:, 1])
            self.print("ag", ag)
            h2 = dg * h2_ + (1 - dg) * h2
            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                h2=h2,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=d_dist.probs,
                ag_probs=a_gate.probs,
                dg_probs=d_gate.probs,
                ag=ag,
                dg=dg,
            )
