import functools
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RecurrentState = namedtuple("RecurrentState", "a a_probs, v")


class Recurrence(nn.Module):
    def __init__(self, n_subtasks: int, embedding_dim):
        super().__init__()
        self.task_embeddings = nn.Embedding(n_subtasks, embedding_dim)
        self.rnn = nn.GRU()  # TODO: args

    def forward(self, inputs, hx):  # TODO: call from Agent
        return self.pack(self.inner_loop(inputs, rnn_hxs=hx))

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, *actions = torch.split(  # TODO: include actions in f and e
            inputs.detach(), [D - sum(self.size_actions)] + self.size_actions, dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)  # TODO write this method
        inputs = inputs._replace(base=inputs.base.view(T, N, *self.obs_shape))

        # build memory
        rnn_inputs = self.task_embeddings(inputs.subtasks)
        M = self.rnn(rnn_inputs)  # TODO fix this return value
        M, M_minus, M_plus, p0 = torch.split(M, 4, dim=-1)  # TODO is 4 correct?

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)  # TODO: write parse_hidden
        for x in hx:
            x.squeeze_(0)
        p = hx.p
        p[new_episode] = p0[new_episode]
        for t in range(T):
            e = self.psi(inputs.base)
            p += F.cosine_similarity(e, M_plus)
            p -= F.cosine_similarity(e, M_minus)
            r = p @ M
            hidden = self.f((r, inputs.base))  # TODO: build this network
            v = self.critic(hidden)  # TODO: build this network
            a_dist = self.actor(hidden)  # TODO: build this network
            a = a_dist.sample()
            yield RecurrentState(a=a, a_probs=a_dist.probs, v=v)
