from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.blocks_world.envs.planner import Actions
from ppo.distributions import FixedCategorical
from ppo.layers import Product, Flatten
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState",
    "search_logits planned_actions planned_states probs actions state value model_loss embed_loss",
)

INF = 1e8


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
        planning_steps,
        planning_horizon,
        embedding_size,
    ):
        super().__init__()
        self.action_size = 1
        self.debug = debug
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())
        self.planning_steps = planning_steps
        self.planning_horizon = planning_horizon
        self.num_options = action_space.n  # TODO: change

        self.state_sizes = RecurrentState(
            search_logits=planning_steps * self.num_options,
            planned_actions=planning_horizon,
            planned_states=planning_horizon * embedding_size,
            probs=action_space.n,
            actions=1,
            state=embedding_size,
            value=1,
            model_loss=1,
            embed_loss=1,
        )
        self.action_sections = Actions(actual=1, searched=planning_steps)

        # networks
        self.embed1 = nn.Embedding(nvec.max(), nvec.max())
        self.embed_options = nn.Embedding(self.num_options, self.num_options)
        self.controller = nn.GRU(embedding_size, hidden_size, num_layers)
        self.embed2 = init_(nn.Linear(hidden_size, embedding_size))
        self.sharpener = nn.Sequential(activation, init_(nn.Linear(hidden_size, 1)))
        self.critic = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, self.num_options))
        )
        self.model = nn.Sequential(
            Product(),
            Flatten(),
            activation,
            init_(nn.Linear(embedding_size * self.num_options, hidden_size)),
            activation,
            init_(nn.Linear(hidden_size, embedding_size)),
        )
        self.eye = nn.Embedding.from_pretrained(torch.eye(self.num_options))

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
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def parse_actions(self, actions):
        return Actions(torch.split(actions, self.action_sections))

    def print(self, t, *args, **kwargs):
        if self.debug:
            if type(t) == torch.Tensor:
                t = (t * 10.0).round() / 10.0
            print(t, *args, **kwargs)

    def redistribute(self, logits, index):
        return logits - torch.inf(device=logits) * self.eye(index)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        embedded = self.embed1(inputs.long().view(-1, D)).view(T, N, -1)
        H, _ = self.controller(embedded)

        actions = self.parse_actions(actions)

        new = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        # i dimensional
        search_logits = hx.search_logits.view(N, self.planning_steps, self.num_options)[
            new
        ]
        P = actions.searched.view(N, self.planning_steps, self.action_sections.planned)[
            new
        ]
        # TODO write wrapper handling many actions
        # TODO write wrapper combining constraints

        # I dimensional
        n_new = torch.sum(new)  # type: int
        I = torch.zeros(n_new, device=new.device)
        planned_actions = hx.planned_actions.long()[new]
        X = hx.planned_states.view(N, self.planning_horizon, -1)[new]
        X[I] = H[0, new]
        logits = torch.zeros(n_new, self.num_options, device=new.device)

        for i in range(P.size(-1)):
            # TODO: prevent cycles?
            x = X[I]

            sharpness = self.sharpener(x)
            values = self.critic(x)
            new = (logits[I] == 0).all(-1)
            search_logits[i][new] = sharpness * values
            old = (logits[I] != 0).any(-1)
            search_logits[i][old] = logits[I][old]
            self.sample_new(P[i], FixedCategorical(logits=search_logits[i]))

            # stack stuff
            logits[I] = search_logits[i] - INF * self.eye(P[i])
            push = values[P[i]] > 0
            pop = values[P[i]] <= 0
            planned_actions[I] = P[i]
            X[I + 1][push] = self.model(
                (X[I].unsqueeze(1), self.embed_action(P[i]).unsqueeze(2))
            )[push]
            I[push] = torch.min(I[push] + 1, self.planning_horizon)
            I[pop] = torch.max(I[pop] - 1, other=0)
            # TODO: somehow add early termination

        # TODO: add obs to recurrence
        for t in range(T):
            model_loss = (
                self.model((inputs[t - 1], self.embed_action(P[t])))
                - inputs[t].detach()
            ) ** 2
            # TODO: predict from start of episode

            # TODO add gate (c) so that obs is not compared every turn
            logits = F.cosine_similarity(inputs[t], X)
            dist = FixedCategorical(logits=self.sharpener(inputs[t]) * logits)
            embed_loss = -dist.entropy()  # maximize distinctions among embedded
            # self.sample_new(
            #     planned_actions[t], dist
            # )  # todo: distinguish actions from planned actions

            action = planned_actions[t]
            yield RecurrentState(
                search_logits=search_logits,
                planned_actions=planned_actions,
                planned_states=X,
                probs=None,
                actions=action,
                state=inputs[t],
                value=search_logits[t, action],
                model_loss=model_loss,
                embed_loss=embed_loss,
            )


"""
Questions:
- prevent cycles
- early termination
- gate network
- controller design
- model loss
"""
