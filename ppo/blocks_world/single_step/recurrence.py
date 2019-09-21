from collections import namedtuple

import numpy as np
from gym.spaces import Box
from torch import nn as nn

from ppo.agent import NNBase
from ppo.distributions import Categorical
from ppo.utils import init, init_normc_, init_

RecurrentState = namedtuple("RecurrentState", "a probs v")
# "planned_probs plan v t state h model_loss"


class Recurrence(nn.Module):
    def __init__(
        self, observation_space, action_space, activation, hidden_size, num_layers
    ):
        super().__init__()
        self.action_size = 1
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())
        self.hidden_size = hidden_size

        self.state_sizes = RecurrentState(a=1, v=1, probs=action_space.n)

        # networks
        # self.embed_action = nn.Embedding(int(action_space.n), int(action_space.n))
        layers = []
        in_size = int(np.prod(nvec.shape))
        for _ in range(num_layers):
            layers += [activation, init_(nn.Linear(in_size, hidden_size))]
            in_size = hidden_size
        self.embed1 = nn.Sequential(*layers)

        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, action_space.n)
        self.continuous = isinstance(action_space, Box)

        self.train()

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, rnn_hxs, masks, action):
        x = self.embed1(inputs)

        dist = self.actor(x)
        self.sample_new(action, dist)

        return RecurrentState(a=action, probs=dist.probs, v=self.critic(x))
