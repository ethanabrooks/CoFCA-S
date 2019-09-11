from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.bandit.bandit
from ppo.distributions import FixedCategorical
from ppo.layers import Concat
from ppo.maze.env import Actions
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a v h a_probs")


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
    ):
        super().__init__()

        self.obs_shape = h, w, d = observation_space.shape
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        layers = []
        in_size = d
        for _ in range(num_layers + 1):
            layers += [
                nn.Conv2d(in_size, hidden_size, kernel_size=3, padding=1),
                activation,
            ]
            in_size = hidden_size
        self.task_embedding = nn.Sequential(*layers)
        self.task_encoder = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.query_generator = nn.Linear(hidden_size, 2 * hidden_size)
        self.terminator = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(h * w))
        self.state_sizes = RecurrentState(a=1, a_probs=h * w * 2, v=1, h=hidden_size)

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
        return ppo.bandit.bandit.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        obs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=-1
        )
        obs = obs[0]

        # build memory
        M = (
            self.task_embedding(
                obs.view(N, *self.obs_shape).permute(2, 0, 1)
            )  # N, hidden_size, h, w
            .view(N, self.hidden_size, -1)  # N, hidden_size, h * w
            .transpose(1, 2)  # N, h * w, hidden_size
        )
        K, _ = self.task_encoder(M.transpose(0, 1))  # h * w, N, hidden_size * 2
        K = K.transpose(0, 1)  # N, h * w, hidden_size * 2

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            k = self.query_generator(h)
            done = self.terminator(h)
            w = (K @ k.unsqueeze(2)).squeeze(2)
            self.print("w")
            self.print(w)
            logits = torch.cat([(1 - done) * w, done * w], dim=-1)
            dist = FixedCategorical(logits=logits)
            self.print("dist")
            self.print(dist.probs)
            self.sample_new(A[t], dist)
            a_size = self.a_one_hots.embedding_dim
            p = self.a_one_hots(A[t] % a_size)
            r = (p.unsqueeze(1) @ M).squeeze(1)
            h = self.gru(r, h)
            yield RecurrentState(a=A[t], v=self.critic(h), h=h, a_probs=dist.probs)
