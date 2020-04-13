from collections import namedtuple
from contextlib import contextmanager
from ppo.control_flow.env import Obs
from ppo.control_flow.recurrence import get_obs_sections

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

import ppo.control_flow.recurrence
from ppo.utils import init_


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


RecurrentState = namedtuple("RecurrentState", "a p w v h h2 a_probs")


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def __init__(self, hidden_size, num_layers, activation, **kwargs):
        super().__init__(
            hidden_size=2 * hidden_size,
            num_layers=num_layers,
            activation=activation,
            **kwargs,
        )
        self.upsilon = init_(nn.Linear(2 * hidden_size, 3))
        self.state_sizes = RecurrentState(
            a=1,
            p=1,
            a_probs=self.state_sizes.a_probs,
            w=len(self.obs_spaces.lines.nvec),
            v=1,
            h=hidden_size,
            h2=hidden_size,
        )

    # noinspection PyProtectedMember
    @contextmanager
    def evaluating(self, eval_obs_space):
        with super().evaluating(eval_obs_space) as self:
            self.state_sizes = self.state_sizes._replace(
                w=len(eval_obs_space.spaces["lines"].nvec)
            )
            yield self

    def set_obs_space(self, obs_space):
        self.obs_spaces = Obs(**obs_space.spaces)
        self.obs_sections = get_obs_sections(self.obs_spaces)
        self.train_lines = len(self.obs_spaces.lines.nvec)

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
        M = self.build_memory(N, T, inputs)
        indices = torch.arange(M.size(1), device=rnn_hxs.device).float()

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        h2 = hx.h2
        w = hx.w
        w[new_episode, 0] = 1
        hx.a[new_episode] = self.n_a - 1
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("w", w)
            obs = self.preprocess_obs(inputs.obs[t])
            r = (w.unsqueeze(1) @ M).squeeze(1)
            x = [obs, r, self.embed_upper(A[t - 1].clone())]
            h_cat = torch.cat([h, h2], dim=-1)
            h_cat2 = self.gru(torch.cat(x, dim=-1), h_cat)
            z = F.relu(self.zeta(h_cat2))

            l = self.upsilon(z).softmax(dim=-1)
            w = batch_conv1d(w, l)

            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)

            h_size = self.hidden_size // 2
            h, h2 = torch.split(h_cat2, [h_size, h_size], dim=-1)

            p = torch.round(w @ indices.unsqueeze(1))
            yield RecurrentState(
                a=A[t], w=w, v=self.critic(z), h=h, h2=h2, a_probs=a_dist.probs, p=p
            )
