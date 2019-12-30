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

        self.d_gate = Categorical(hidden_size, 2)
        self.a_gate = Categorical(hidden_size, 2)
        self.actor = Categorical(hidden_size, self.n_a - 1)
        self.state_sizes = self.state_sizes._replace(
            a_probs=self.state_sizes.a_probs - 1
        )
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
        return (
            # self.conv_hidden_size
            self.hidden_size
            + self.train_lines * self.encoder_hidden_size
        )

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

        G, H = self.task_encoder(M)
        H = H.transpose(0, 1).reshape(N, -1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        hx.a[new_episode] = self.n_a - 1
        ag_probs = hx.ag_probs
        ag_probs[new_episode, 1] = 1
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()
        AG = torch.cat([actions[:, :, 2], hx.ag.view(1, N)], dim=0).long()

        for t in range(T):
            if self.use_conv:
                obs = self.conv(inputs.obs[t]).view(N, -1)
            else:
                obs = (
                    self.conv(inputs.obs[t].permute(0, 2, 3, 1))
                    .view(N, -1, self.conv_hidden_size)
                    .max(dim=1)
                    .values
                )
            # x = [obs, M.view(N, -1), self.embed_action(A[t - 1].clone())]
            x = [M.view(N, -1), self.embed_action(A[t - 1].clone())]
            X = torch.cat(x, dim=-1)
            h = self.gru(X, h)
            z = F.relu(self.zeta(h))
            a_gate = self.a_gate(z)
            self.sample_new(AG[t], a_gate)
            ag = AG[t].unsqueeze(-1).float()
            # a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            # self.sample_new(A[t], a_dist)
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)
            self.print("ag prob", torch.round(100 * a_gate.probs[:, 1]))
            self.print("ag", ag)

            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                h2=hx.h2,
                p=hx.p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=hx.d_probs,
                ag_probs=a_gate.probs,
                dg_probs=hx.dg_probs,
                ag=ag,
                dg=hx.dg,
            )
