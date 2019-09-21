from collections import namedtuple

import numpy as np
import torch
from gym.spaces import Box
from torch import nn as nn

from ppo.distributions import Categorical, FixedCategorical
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a probs v state h ")
# "planned_probs plan v t state h model_loss"


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_embedding_layers,
        num_model_layers,
        embedding_size,
        activation,
        planning_steps,
    ):
        num_inputs = int(np.prod(observation_space.shape))
        super().__init__()
        self.action_size = planning_steps

        na = action_space.nvec.max()
        self.state_sizes = RecurrentState(
            a=planning_steps,
            v=1,
            probs=planning_steps * na,
            state=embedding_size,
            h=hidden_size * num_model_layers,
        )

        # networks
        self.embed_action = nn.Embedding(int(na), int(na))
        layers = []
        in_size = num_inputs
        for _ in range(num_embedding_layers):
            layers += [activation, init_(nn.Linear(in_size, hidden_size))]
            in_size = hidden_size
        self.embed1 = nn.Sequential(*layers)
        self.embed2 = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, embedding_size))
        )
        self.model = nn.GRU(
            embedding_size + self.embed_action.embedding_dim,
            hidden_size,
            num_model_layers,
        )

        self.critic = init_(nn.Linear(embedding_size, 1))
        self.actor = init_(nn.Linear(embedding_size, na))
        self.train()

    def print(self, t, *args, **kwargs):
        if self.debug:
            if type(t) == torch.Tensor:
                t = (t * 10.0).round() / 10.0
            print(t, *args, **kwargs)

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, rnn_hxs):
        return self.pack(self.inner_loop(inputs, rnn_hxs))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

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

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, action = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        new = torch.all(rnn_hxs == 0, dim=-1)
        if new.any():
            assert new.all()
            h = (
                hx.h.view(N, self.model.num_layers, self.model.hidden_size)
                .transpose(0, 1)
                .contiguous()
            )

            A = action.long()
            first_state = state = self.embed2(self.embed1(inputs[0]))
            probs = []
            for t in range(self.action_size):
                dist = FixedCategorical(logits=self.actor(state))
                self.sample_new(A[0, :, t], dist)
                probs.append(dist.probs)
                model_input = torch.cat(
                    [state, self.embed_action(A[0, :, t].clone())], dim=-1
                )
                hn, h = self.model(model_input.unsqueeze(0), h)
                state = self.embed2(hn.squeeze(0))
        v = self.critic(first_state)
        probs = torch.stack(probs, dim=1)
        for t in range(T):
            yield RecurrentState(a=A[t], probs=probs, v=v, state=state, h=hx.h)
