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
        self.obs_shape = h, w = self.obs_spaces.rewards.shape
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size
        self.na = action_space.n

        # networks
        layers = [Concat(dim=-1)]
        in_size = hidden_size * 2
        for i in range(num_layers):
            layers += [init_(nn.Linear(in_size, hidden_size)), activation]
            in_size = hidden_size
        self.f = nn.Sequential(*layers)
        self.actor = Categorical(in_size, self.na)
        self.critic = init_(nn.Linear(in_size, 1))
        self.register_buffer("state_indexer", torch.tensor([w, 1.0]))
        self.S = nn.Embedding(h * w, hidden_size)
        _S = nn.Parameter(torch.normal(0, 1, size=(h * w, hidden_size)))
        S = self.S.weight
        A = nn.Parameter(torch.normal(0, 1, size=(int(action_space.n), hidden_size)))
        SA = nn.Parameter(S.unsqueeze(-1) + A.T.unsqueeze(0))
        self.T = nn.Parameter(S @ SA)
        # SA = nn.Parameter(
        #     torch.normal(0, 1, size=(h * w, hidden_size, int(action_space.n)))
        # )
        # T = torch.normal(
        #     torch.zeros(h * w, h * w, int(action_space.n)),
        #     torch.ones(h * w, h * w, int(action_space.n)),
        # )
        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(int(h * w)))
        self.state_sizes = RecurrentState(
            a=1, a_probs=action_space.n, v=1, h=in_size, estimated_values=h * w
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
        ns = self.S.num_embeddings
        na = self.na
        obs, action = torch.split(inputs.detach(), [D - 1, 1], dim=-1)
        obs = self.parse_inputs(obs)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        A = torch.cat([action, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)
        for t in range(T):
            values = torch.zeros_like(obs.rewards[t])
            for _ in range(self.time_limit):
                P = self.T.softmax(dim=1)
                EV = (values @ P.view(ns, -1)).view(N, ns, na)
                maxEV = EV.max(dim=-1)
                values = obs.rewards[t] + maxEV.values

            state_idx = (obs.state[t] @ self.state_indexer).long().unsqueeze(-1)
            next_state_idx = maxEV.indices.gather(dim=-1, index=state_idx)
            s1 = self.S(state_idx).squeeze(1)
            s2 = self.S(next_state_idx).squeeze(1)
            h = self.f((s1, s2))

            dist = self.actor(h)
            # self.sample_new(A[t], dist)

            A[t] = torch.cat(
                [
                    next_state_idx // self.size - state_idx // self.size,
                    next_state_idx % self.size - state_idx % self.size,
                ],
                dim=-1,
            )
            yield RecurrentState(
                a=A[t],
                a_probs=dist.probs,
                v=self.critic(h),
                h=h,
                estimated_values=values,
            )
