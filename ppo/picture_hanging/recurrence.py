from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.oh_et_al
from ppo.distributions import FixedCategorical, Categorical, DiagGaussian
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a probs loc scale v h p")


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
        kernel_radius,
        debug,
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.conv = nn.Sequential(
            init_(
                nn.Conv1d(
                    1,
                    hidden_size,
                    kernel_size=kernel_radius * 2 + 1,
                    padding=kernel_radius,
                ),
                "conv1d",
            ),
            activation,
        )
        self.gru0 = nn.GRU(hidden_size, hidden_size, num_layers)
        self.gru1 = nn.GRU(hidden_size, hidden_size, num_layers)
        self.critic = init_(nn.Linear(hidden_size, 1))
        # self.actor = DiagGaussian(
        #     hidden_size, 1, limits=(action_space.low.item(), action_space.high.item())
        # )
        self.actor = Categorical(hidden_size, action_space.n)
        self.state_sizes = RecurrentState(
            a=1,
            probs=action_space.n,
            loc=1,
            scale=1,
            p=1,
            v=1,
            h=num_layers * hidden_size,
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
        return ppo.oh_et_al.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            torch.set_printoptions(precision=2, sci_mode=False)
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        device = inputs.device
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        # build memory
        H = self.conv(inputs[0].unsqueeze(1))
        M, hn = self.gru0(H.permute(2, 0, 1))
        M = M.transpose(0, 1)

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = (
            hx.h.view(N, self.gru1.num_layers, self.gru1.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        p = hx.p.long().squeeze(1)
        A = actions[:, :, 0].long()
        R = torch.arange(N, device=device)

        for t in range(T):
            r = M[R, p]
            hn, h = self.gru1(r.unsqueeze(0), h)  # (seq_len, batch, input_size)
            v = self.critic(hn.squeeze(0))
            dist = self.actor(hn.squeeze(0))
            self.sample_new(A[t], dist)
            p = p + 1
            self.print(v)
            # self.print(p)
            yield RecurrentState(
                a=A[t],
                probs=dist.probs,
                loc=hx.loc,  # TODO
                scale=hx.scale,  # TODO
                v=v,
                h=h.transpose(0, 1),
                p=p,
            )
