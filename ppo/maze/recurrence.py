from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

import ppo.oh_et_al
import ppo.bandit.bandit
from ppo.distributions import FixedCategorical
from ppo.layers import Concat
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a v h a_probs p")


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
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.task_embedding = nn.Conv2d(d, hidden_size, kernel_size=3, padding=1)
        self.task_encoder = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        # f
        layers = [Concat(dim=-1)]
        in_size = self.obs_sections.condition + hidden_size
        for _ in range(num_layers + 1):
            layers.extend([nn.Linear(in_size, hidden_size), activation])
            in_size = hidden_size
        self.f = nn.Sequential(*layers)

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = nn.Linear(hidden_size, hidden_size)
        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(action_space.n))
        self.state_sizes = RecurrentState(
            a=1, a_probs=(action_space.n), p=action_space.n, v=1, h=hidden_size
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
        obs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        # build memory
        embeddings = self.task_embedding(obs.view(*self.obs_shape))
        K = self.task_encoder()
        M = self.embeddings(lines.view(-1)).view(
            *lines.shape, self.hidden_size
        )  # n_batch, n_lines, hidden_size

        forward_input = M.transpose(0, 1)  # n_lines, n_batch, hidden_size
        backward_input = forward_input.flip((0,))
        keys = []
        for i in range(len(forward_input)):
            keys_per_i = []
            if i > 0:
                backward, _ = self.task_encoder(backward_input[:i])
                keys_per_i.append(backward.flip((0,)))
            if i < len(forward_input):
                forward, _ = self.task_encoder(forward_input[i:])
                keys_per_i.append(forward)
            keys.append(torch.cat(keys_per_i).transpose(0, 1))  # put batch dim first
        K = torch.stack(keys, dim=1)  # put from dim before to dim
        K, C = torch.split(K, [self.hidden_size, 1], dim=-1)
        K = K.sum(dim=1)
        C = C.squeeze(dim=-1)
        self.print("C")
        self.print(C)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        p = hx.p
        p[new_episode, 0] = 1
        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            r = (p.unsqueeze(1) @ M).squeeze(1)
            h = self.gru(self.f((obs.condition[t], r)), h)
            k = self.actor(h)
            c = (p.unsqueeze(1) @ C).squeeze(1)
            self.print("c")
            self.print(c)
            w = (K @ k.unsqueeze(2)).squeeze(2)
            self.print("w")
            self.print(w)
            self.print("w * c")
            self.print(w * c)
            dist = FixedCategorical(logits=w * c)
            self.print("dist")
            self.print(dist.probs)
            self.sample_new(A[t], dist)
            p = self.a_one_hots(A[t])
            yield RecurrentState(a=A[t], v=self.critic(h), h=h, a_probs=dist.probs, p=p)
