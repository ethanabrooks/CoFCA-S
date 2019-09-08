from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.bandit.bandit
from ppo.distributions import FixedCategorical, Categorical
from ppo.layers import Concat
from ppo.maze.env import Actions
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a v h a_probs p p_probs")


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
        self.obs_shape = d, h, w = observation_space.shape
        self.action_size = 2
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
        self.actor = Categorical(hidden_size, action_space.spaces["a"].n)
        self.query_generator = nn.Linear(hidden_size, 2 * hidden_size)
        self.terminator = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.p_one_hots = nn.Embedding.from_pretrained(torch.eye(int(h * w)))
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_space.spaces["a"].n,
            p=1,
            p_probs=h * w,
            v=1,
            h=hidden_size,
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
        return ppo.bandit.bandit.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        obs, actions, attentions = torch.split(inputs.detach(), [D - 2, 1, 1], dim=-1)
        obs = obs[0]

        # build memory
        M = (
            self.task_embedding(obs.view(N, *self.obs_shape))  # N, hidden_size, h, w
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
        P = torch.cat([attentions, hx.p.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            k = self.query_generator(h)
            w = (K @ k.unsqueeze(2)).squeeze(2)
            p_dist = FixedCategorical(logits=w)
            self.sample_new(P[t], p_dist)
            p = self.p_one_hots(P[t])
            r = (p.unsqueeze(1) @ M).squeeze(1)
            h = self.gru(r, h)
            a_dist = self.actor(h)
            self.sample_new(A[t], a_dist)
            yield RecurrentState(
                a=A[t],
                a_probs=a_dist.probs,
                v=self.critic(h),
                h=h,
                p_probs=p_dist.probs,
                p=P[t],
            )
