from collections import namedtuple

import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

import ppo.control_flow.recurrence
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_
import numpy as np


RecurrentState = namedtuple("RecurrentState", "a v h a_probs")


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def __init__(
        self, hidden_size, num_layers, activation, conv_hidden_size, use_conv, **kwargs
    ):
        self.conv_hidden_size = conv_hidden_size
        self.use_conv = use_conv
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            **kwargs
        )
        self.action_size = 1
        d = self.obs_spaces.obs.shape[0]
        self.conv = nn.Sequential(init_(nn.Linear(d, conv_hidden_size)), nn.ReLU())
        self.actor = Categorical(hidden_size, self.n_a - 1)
        self.state_sizes = self.state_sizes._replace(
            a_probs=self.state_sizes.a_probs - 1
        )
        self.state_sizes = RecurrentState(a=1, v=1, h=hidden_size, a_probs=self.n_a)
        line_nvec = torch.tensor(self.obs_spaces.lines.nvec[0, :-1])
        offset = F.pad(line_nvec.cumsum(0), [1, 0])
        self.register_buffer("offset", offset)

    def build_embed_task(self, hidden_size):
        return nn.EmbeddingBag(self.obs_spaces.lines.nvec[0].sum(), hidden_size)

    @property
    def gru_in_size(self):
        return self.conv_hidden_size + 2 * self.encoder_hidden_size + self.hidden_size

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
        hx = self.parse_hidden(rnn_hxs)

        obs = (
            self.conv(inputs.obs.permute(0, 2, 3, 1))
            .view(N, -1, self.conv_hidden_size)
            .max(dim=1)
            .values
        )
        x = torch.cat([M, obs, self.embed_action(A[t - 1].clone())])
