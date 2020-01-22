from collections import namedtuple
from contextlib import contextmanager

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
    return FixedCategorical(probs=g * new + (1 - g) * old)


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
        **kwargs,
    ):
        self.gate_h = gate_h
        self.nl_2 = nl_2
        self.conv_hidden_size = conv_hidden_size
        self.use_conv = use_conv
        super().__init__(
            hidden_size=2 * hidden_size,
            num_layers=num_layers,
            activation=activation,
            **kwargs,
        )
        self.upsilon = init_(nn.Linear(2 * hidden_size, 3))
        self.gate_coef = gate_coef
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
            if kernel_size < 4:
                layers += [
                    nn.Conv2d(
                        conv_hidden_size,
                        conv_hidden_size,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.ReLU(),
                ]
            self.conv = nn.Sequential(*layers)
        else:
            self.conv = nn.Sequential(init_(nn.Linear(d, conv_hidden_size)), nn.ReLU())

        self.d_gate = nn.Sequential(init_(nn.Linear(2 * hidden_size, 1)), nn.Sigmoid())
        self.a_gate = nn.Sequential(init_(nn.Linear(2 * hidden_size, 1)), nn.Sigmoid())
        self.state_sizes = RecurrentState(
            **self.state_sizes._replace(p=self.train_lines)
            ._replace(h=hidden_size)
            ._asdict(),
            h2=hidden_size,
            ag_probs=0,
            dg_probs=0,
            ag=1,
            dg=1,
        )

        line_nvec = torch.tensor(self.obs_spaces.lines.nvec[0, :-1])
        offset = F.pad(line_nvec.cumsum(0), [1, 0])
        self.register_buffer("offset", offset)

    # noinspection PyProtectedMember
    @contextmanager
    def evaluating(self, eval_obs_space):
        with super().evaluating(eval_obs_space) as self:
            self.state_sizes = self.state_sizes._replace(
                p=len(eval_obs_space.spaces["lines"].nvec)
            )
            yield self

    def build_embed_task(self, hidden_size):
        return nn.EmbeddingBag(self.obs_spaces.lines.nvec[0].sum(), hidden_size)

    @property
    def gru_in_size(self):
        return self.hidden_size + self.conv_hidden_size + self.encoder_hidden_size

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
        lines = inputs.lines.view(T, N, *self.obs_spaces.lines.shape)
        lines = lines.long()[0, :, :] + self.offset
        M = self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            *lines.shape[:2], self.encoder_hidden_size
        )  # n_batch, n_lines, hidden_size

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        h2 = hx.h2
        p = hx.p
        p[new_episode, 0] = 1
        hx.a[new_episode] = self.n_a - 1
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()

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
            r = (p.unsqueeze(1) @ M).squeeze(1)
            x = [obs, r, self.embed_action(A[t - 1].clone())]
            h_cat = torch.cat([h, h2], dim=-1)
            h_cat2 = self.gru(torch.cat(x, dim=-1), h_cat)
            z = F.relu(self.zeta(h_cat2))
            d_gate = self.d_gate(z)
            a_gate = self.a_gate(z)

            l = self.upsilon(z).softmax(dim=-1)
            p_ = batch_conv1d(p, l)
            p = d_gate * p_ + (1 - d_gate) * p

            a_probs = self.actor(z).probs
            old = torch.zeros_like(a_probs).scatter(1, A[t - 1].unsqueeze(1), 1)
            a_dist = gate(a_gate, a_probs, old)
            self.sample_new(A[t], a_dist)
            # self.print("ag prob", torch.round(100 * a_gate.probs[:, 1]))

            h_size = self.hidden_size // 2
            h_, h2 = torch.split(h_cat2, [h_size, h_size], dim=-1)
            h = d_gate * h_ + (1 - d_gate) * h_

            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                h2=h2,
                p=p,
                a_probs=a_dist.probs,
                d=hx.d,
                d_probs=hx.d_probs,
                ag_probs=hx.ag_probs,
                dg_probs=hx.dg_probs,
                ag=hx.ag,
                dg=hx.dg,
            )
