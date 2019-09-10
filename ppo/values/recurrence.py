from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.bandit.bandit
from ppo.distributions import FixedCategorical, Categorical
from ppo.layers import Concat
from ppo.maze.env import Actions
from ppo.utils import init_
import numpy as np

RecurrentState = namedtuple("RecurrentState", "a a_probs v h estimated_values")


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
        time_limit,
    ):
        super().__init__()
        self.time_limit = time_limit
        self.obs_spaces = ppo.values.env.Obs(**observation_space.spaces)
        self.obs_sections = ppo.values.env.Obs(
            *[int(np.prod(s.shape)) for s in self.obs_spaces]
        )
        self.obs_shape = h, w = list(map(int, self.obs_spaces.rewards.shape))
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.gru = nn.GRU(hidden_size, hidden_size)
        layers = []
        in_size = hidden_size * 2
        for i in range(num_layers):
            layers += [
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=action_space.n if i == num_layers - 1 else hidden_size,
                    kernel_size=1,
                ),
                activation,
            ]
            in_size = hidden_size
        self.emb = nn.Sequential(*layers)

        S = torch.normal(
            torch.zeros(h * w, hidden_size), torch.ones(h * w, hidden_size)
        )
        A = torch.normal(
            torch.zeros(int(action_space.n), hidden_size),
            torch.ones(int(action_space.n), hidden_size),
        )
        self.S = nn.Parameter(S)
        self.A = nn.Parameter(A)
        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(int(h * w)))
        self.state_sizes = RecurrentState(
            a=1, a_probs=action_space.n, v=1, h=hidden_size, estimated_values=h * w
        )

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
        return ppo.values.env.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        obs, action = torch.split(inputs.detach(), [D - 1, 1], dim=-1)
        obs = self.parse_inputs(obs)
        hx = self.parse_hidden(rnn_hxs)
        K, _ = self.gru(self.S.unsqueeze(1))
        K = K.squeeze(1)

        for t in range(T):
            values = torch.zeros_like(obs.rewards[t])
            for _ in range(self.time_limit):
                emb_input = self.S.unsqueeze(0) + self.A.unsqueeze(1)
                q = self.emb(emb_input.permute(2, 0, 1)).permute(2, 0, 1)
                P = (K @ q).softmax(dim=1)
                EV = values @ P
                values = obs.rewards[t] + EV.max(dim=-1).values.t()

            yield RecurrentState(
                a=hx.a[t],
                a_probs=hx.a_probs[t],
                v=hx.v[t],
                h=hx.h[t],
                estimated_values=values,
            )
