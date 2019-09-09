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

RecurrentState = namedtuple("RecurrentState", "a a_probs v h M")


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
        num_conv_layers,
        debug,
    ):
        super().__init__()
        self.obs_spaces = ppo.random_rewards.env.Obs(**observation_space.spaces)
        self.obs_sections = ppo.random_rewards.env.Obs(
            *[int(np.prod(s.shape)) for s in self.obs_spaces]
        )
        self.obs_shape = h, w = self.obs_spaces.rewards.shape
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        layers = []
        in_size = 1
        for _ in range(num_conv_layers + 1):
            layers += [
                init_(
                    nn.Conv2d(in_size, hidden_size, kernel_size=3, padding=1), "Conv2d"
                ),
                activation,
            ]
            in_size = hidden_size
        self.task_embedding = nn.Sequential(*layers)
        layers = [Concat(dim=-1)]
        in_size = self.hidden_size + 1
        for _ in range(num_layers + 1):
            layers += [init_(nn.Linear(in_size, hidden_size)), activation]
            in_size = hidden_size
        self.f = nn.Sequential(*layers)
        self.task_encoder = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.go = init_(nn.Linear(hidden_size, self.hidden_size * 3 + 1))
        self.write = init_(nn.Linear(hidden_size, self.hidden_size + 1))

        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(int(h * w)))
        self.state_sizes = RecurrentState(
            a=1, a_probs=action_space.n, v=1, h=hidden_size, M=h * w * hidden_size
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
        return ppo.random_rewards.env.Obs(
            *torch.split(inputs, self.obs_sections, dim=-1)
        )

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        obs, action = torch.split(inputs.detach(), [D - 1, 1], dim=-1)
        obs = self.parse_inputs(obs)

        # build memory
        M0 = (
            self.task_embedding(
                obs.rewards[0].view(N, 1, *self.obs_shape)
            )  # N, hidden_size, h, w
            .view(N, self.hidden_size, -1)  # N, hidden_size, h * w
            .transpose(1, 2)  # N, h * w, hidden_size
        )
        K, _ = self.task_encoder(M0.transpose(0, 1))  # h * w, N, hidden_size * 2
        K = K.transpose(0, 1)  # N, h * w, hidden_size * 2

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        M = hx.M.view(N, -1, self.hidden_size)
        new_episode = torch.all(rnn_hxs == 0)
        M[new_episode] = M0[new_episode]
        A = torch.cat([action, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            qk, qm, b = torch.split(
                self.go(h), [self.hidden_size * 2, self.hidden_size, 1], dim=-1
            )
            wk = F.softmax((K @ qk.unsqueeze(2)).squeeze(2), dim=-1)
            wm = F.softmax((M @ qm.unsqueeze(2)).squeeze(2), dim=-1)
            b = b.sigmoid()
            probs = b * wk + (1 - b) * wm
            dist = FixedCategorical(probs=probs)
            self.sample_new(A[t], dist)
            a = self.a_one_hots(A[t])
            r = (a.unsqueeze(1) @ M).squeeze(1)
            h = self.gru(self.f((r, obs.go[t])), h)
            v, o = torch.split(self.write(h), [self.hidden_size, 1], dim=-1)
            w = (o.sigmoid() * a).unsqueeze(-1)
            M = w * v.unsqueeze(1) + (1 - w) * M
            yield RecurrentState(a=A[t], a_probs=dist.probs, v=self.critic(h), h=h, M=M)
