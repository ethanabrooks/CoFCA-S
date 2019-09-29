from collections import namedtuple
import ppo.oh_et_al

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from ppo.distributions import FixedCategorical, Categorical
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
        self.obs_spaces = ppo.oh_et_al.Obs(**observation_space.spaces)
        self.obs_sections = ppo.oh_et_al.Obs(
            *[int(np.prod(s.shape)) for s in self.obs_spaces]
        )
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size
        self.register_buffer(
            "subtask_floor",
            torch.tensor([0] + list(self.obs_spaces.subtasks.nvec[0, :-1])),
        )

        # networks
        self.embeddings = nn.EmbeddingBag(
            int(self.obs_spaces.subtasks.nvec[0].sum()), hidden_size
        )

        # f
        in_size = self.obs_sections.obs + hidden_size
        self.gru = nn.GRU(in_size, hidden_size, num_layers)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.phi_update = init_(nn.Linear(hidden_size, 1))
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_space.n,
            p=self.obs_spaces.subtasks.shape[0],
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
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)

        # build memory
        n_subtasks = self.obs_spaces.subtasks.shape[0]
        subtasks = inputs.subtasks[0].long().view(N * n_subtasks, 2)
        M = self.embeddings(subtasks + self.subtask_floor).view(
            N, n_subtasks, self.hidden_size
        )
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = (
            hx.h.view(N, self.gru.num_layers, self.gru.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        p = hx.p
        p[new_episode, 0] = 1
        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            r = (p.unsqueeze(1) @ M).squeeze(1)
            gru_inputs = torch.cat([inputs.obs[t], r], dim=-1).unsqueeze(0)
            hn, h = self.gru(gru_inputs, h)
            c = self.phi_update(hn.squeeze(0)).sigmoid()
            p = (1 - c) * p + c * torch.roll(p, shifts=1)
            dist = FixedCategorical(p)
            self.sample_new(A[t], dist)
            yield RecurrentState(
                a=A[t],
                v=self.critic(hn.squeeze(0)),
                h=h.transpose(0, 1),
                a_probs=dist.probs,
                p=p,
            )
