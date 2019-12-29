from collections import namedtuple

import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

import ppo.control_flow.recurrence
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_
import numpy as np

RecurrentState = namedtuple(
    "RecurrentState", "a d ag dg p v h h2 a_probs d_probs ag_probs dg_probs"
)


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def __init__(
        self,
        hidden_size,
        gate_coef,
        num_layers,
        activation,
        conv_hidden_size,
        use_conv,
        kernel_size,
        nl_2,
        gate_h,
        **kwargs
    ):
        self.gate_h = gate_h
        self.nl_2 = nl_2
        self.conv_hidden_size = conv_hidden_size
        self.use_conv = use_conv
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            **kwargs
        )
        self.gate_coef = gate_coef
        self.action_size = 4
        d = self.obs_spaces.obs.shape[0]
        if use_conv:
            layers = [
                nn.Conv2d(
                    d,
                    conv_hidden_size,
                    kernel_size=kernel_size,
                    stride=2 if kernel_size == 2 else 1,
                    padding=0,
                ),
                nn.ReLU(),
            ]
            # if kernel_size < 4:
            # layers += [
            # nn.Conv2d(
            # conv_hidden_size,
            # conv_hidden_size,
            # kernel_size=2,
            # stride=2,
            # padding=0,
            # ),
            # nn.ReLU(),
            # ]
            self.conv = nn.Sequential(*layers)
        else:
            self.conv = nn.Sequential(init_(nn.Linear(d, conv_hidden_size)), nn.ReLU())

        self.d_gate = Categorical(hidden_size, 2)
        self.a_gate = Categorical(hidden_size, 2)
        self.state_sizes = RecurrentState(
            **self.state_sizes._asdict(),
            h2=hidden_size,
            ag_probs=2,
            dg_probs=2,
            ag=1,
            dg=1
        )
        ones = torch.ones(1, dtype=torch.long)
        self.register_buffer("ones", ones)
        line_nvec = torch.tensor(self.obs_spaces.lines.nvec[0, :-1])
        offset = F.pad(line_nvec.cumsum(0), [1, 0])
        self.register_buffer("offset", offset)

    def build_embed_task(self, hidden_size):
        return nn.EmbeddingBag(self.obs_spaces.lines.nvec[0].sum(), hidden_size)

    @property
    def gru_in_size(self):
        in_size = self.hidden_size + self.conv_hidden_size
        if self.no_pointer:
            return in_size + 2 * self.hidden_size
        else:
            return in_size + self.encoder_hidden_size

    @staticmethod
    def eval_lines_space(n_eval_lines, train_lines_space):
        return spaces.MultiDiscrete(
            np.repeat(train_lines_space.nvec[:1], repeats=n_eval_lines, axis=0)
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
        lines = inputs.lines.view(T, N, *self.obs_spaces.lines.shape)
        lines = lines.long()[0, :, :] + self.offset
        M = self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            *lines.shape[:2], self.encoder_hidden_size
        )  # n_batch, n_lines, hidden_size

        rolled = []
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
            B = bb = self.beta(G).sigmoid()
            # arange = torch.zeros(6).float()
            # arange[0] = 1
            # arange[1] = 1
            # B[:, :, :, 0] = 0  # arange.view(1, 1, -1, 1)
            # B[:, :, :, 1] = 1
            f, b = torch.unbind(B, dim=3)
            B = torch.stack([f, b.flip(2)], dim=-2)
            B = B.view(nl, N, 2 * nl, self.ne)
            B = (1 - last).flip(2) * B  # this ensures the first B is 0
            zero_last = (1 - last) * B
            B = zero_last + last  # this ensures that the last B is 1
            rolled = torch.roll(zero_last, shifts=1, dims=2)
            C = torch.cumprod(1 - rolled, dim=2)
            P = B * C
            P = P.view(nl, N, nl, 2, self.ne)
            f, b = torch.unbind(P, dim=3)
            half = b.size(2)
            P = torch.cat([b.flip(2), f], dim=2)

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
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()
        AG = torch.cat([actions[:, :, 2], hx.ag.view(1, N)], dim=0).long()
        DG = torch.cat([actions[:, :, 3], hx.dg.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            if self.use_conv:
                obs = self.conv(inputs.obs[t]).view(N, -1)
            else:
                obs = (
                    self.conv(inputs.obs[t].permute(0, 2, 3, 1))
                    .view(N, -1, self.conv_hidden_size)
                    .max(dim=1)
                    .values
                )
            x = [
                obs,
                H.sum(0) if self.no_pointer else M[R, p],
                self.embed_action(A[t - 1].clone()),
            ]

            h2 = self.gru(torch.cat(x, dim=-1), h2)
            z = F.relu(self.zeta(h))
            a_gate = self.a_gate(z)
            self.sample_new(AG[t], a_gate)
            ag = AG[t].unsqueeze(-1).float()
            a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            self.sample_new(A[t], a_dist)
            self.print("ag prob", torch.round(100 * a_gate.probs[:, 1]))
            self.print("ag", ag)

            h_ = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.zeta(h_))
            u = self.upsilon(z).softmax(dim=-1)
            # self.print("bb", torch.round(100 * bb[p, R, :, 0]))
            self.print("u", torch.round(100 * u))
            w = P[p, R]
            d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)
            d_gate = self.d_gate(z)
            self.sample_new(DG[t], d_gate)
            dg = DG[t].unsqueeze(-1).float()
            self.print("dg prob", torch.round(100 * d_gate.probs[:, 1]))
            self.print("dg", dg)
            d_dist = gate(dg, d_probs, ones * nl)
            self.print("d_probs", torch.round(100 * d_probs)[:, nl:])
            self.sample_new(D[t], d_dist)
            p = p + D[t].clone() - nl
            p = torch.clamp(p, min=0, max=nl - (2 if self.nl_2 else 1))

            if self.gate_h:
                h = dg * h_ + (1 - dg) * h
            else:
                h = h_
            # h2 = h2_

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
