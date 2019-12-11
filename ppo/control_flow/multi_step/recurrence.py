import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.control_flow.recurrence
from ppo.distributions import FixedCategorical
from ppo.utils import init_
from ppo.control_flow.recurrence import RecurrentState


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        d = self.obs_spaces.obs.shape[0]
        self.conv = nn.Sequential(
            nn.BatchNorm2d(d),
            nn.Conv2d(d, hidden_size, kernel_size=3, padding=1),
            nn.MaxPool2d(self.obs_spaces.obs.shape[1:]),
            nn.ReLU(),
        )
        self.p_gate = nn.Sequential(init_(nn.Linear(hidden_size, 1)), nn.Sigmoid())
        self.a_gate = nn.Sequential(init_(nn.Linear(hidden_size, 1)), nn.Sigmoid())

    @property
    def gru_in_size(self):
        if self.no_pointer:
            in_size = 3 * self.hidden_size
        elif self.include_action:
            in_size = 2 * self.hidden_size
        else:
            in_size = self.hidden_size
        return in_size + self.obs_sections.obs

    def inner_loop(self, inputs, rnn_hxs):
        T, N, dim = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)
        # inputs = inputs._replace(obs=inputs.obs.view(T, N, *self.obs_spaces.obs.shape))

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
        Z = torch.zeros_like(R)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            # obs = self.conv(inputs.obs[t]).view(N, -1)
            x = [inputs.obs[t], H.sum(0) if self.no_pointer else M[R, p]]
            if self.no_pointer or self.include_action:
                x += [self.embed_action(A[t - 1].clone())]
            h = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.zeta(h))

            def gate(gate, new, old):
                old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
                return FixedCategorical(probs=gate * new + (1 - gate) * old)

            # a_dist = gate(self.a_gate(z), self.actor(z).probs, A[t - 1])
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)
            u = self.upsilon(z).softmax(dim=-1)
            self.print("o", torch.round(10 * u))
            w = P[p, R]
            half1 = w.size(1) // 2
            self.print(torch.round(10 * w)[0, half1:])
            self.print(torch.round(10 * w)[0, :half1])
            d_dist = FixedCategorical(probs=((w @ u.unsqueeze(-1)).squeeze(-1)))
            # p_probs = torch.round(p_dist.probs * 10).flatten()
            self.sample_new(D[t], d_dist)
            half = d_dist.probs.size(-1) // 2
            p = p + D[t].clone() - half
            p = torch.clamp(p, min=0, max=nl - 1)
            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=d_dist.probs,
            )
