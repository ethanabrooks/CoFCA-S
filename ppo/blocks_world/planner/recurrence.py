from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import FixedCategorical
from ppo.layers import Flatten
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState",
    "values search_probs search_options planned_options model_loss embed_loss h x a v",
)
Actions = namedtuple("Actions", "actual searched")

INF = 1e8


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
        embedding_size,
        num_model_layers,
        num_embedding_layers,
    ):
        super().__init__()
        self.action_size = planning_steps + 1
        self.debug = debug
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())
        self.planning_steps = planning_steps
        self.embedding_size = embedding_size
        self.num_options = action_space.nvec.max().item()  # TODO: change
        self.hidden_size = hidden_size
        self.num_model_layers = num_model_layers

        self.state_sizes = RecurrentState(
            values=planning_steps * self.num_options,
            search_probs=planning_steps * self.num_options,
            search_options=planning_steps,
            planned_options=planning_steps,
            model_loss=1,
            embed_loss=1,
            h=num_model_layers * hidden_size,
            x=hidden_size,
            a=1,
            v=1,
        )
        self.action_sections = Actions(actual=1, searched=planning_steps)

        # networks
        self.embed_options = nn.Embedding(self.num_options, self.num_options)

        layers = [nn.Embedding(nvec.max(), nvec.max()), Flatten()]
        in_size = int(nvec.max() * np.prod(nvec.shape))
        for _ in range(num_embedding_layers):
            layers += [activation, init_(nn.Linear(in_size, hidden_size))]
            in_size = hidden_size
        self.embed1 = nn.Sequential(*layers)
        self.embed2 = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, embedding_size))
        )
        self.model = nn.GRU(
            embedding_size + self.num_options, hidden_size, num_model_layers
        )
        self.sharpener = nn.Sequential(activation, init_(nn.Linear(embedding_size, 1)))
        self.critic = nn.Sequential(
            activation, init_(nn.Linear(embedding_size, self.num_options))
        )

        self.eye = nn.Embedding.from_pretrained(torch.eye(self.num_options))
        self.register_buffer("one", torch.tensor(1))
        self.register_buffer("zero", torch.tensor(0))

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
                # print(name, x.shape, view.shape)
                yield x.view(*x.shape[:2], -1)

        hx = torch.cat(list(pack()), dim=-1)
        return hx, hx[-1:]

    def parse_inputs(self, inputs: torch.Tensor):
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def parse_actions(self, actions):
        return Actions(*torch.split(actions, self.action_sections, dim=-1))

    def print(self, t, *args, **kwargs):
        if self.debug:
            if type(t) == torch.Tensor:
                t = (t * 10.0).round() / 10.0
            print(t, *args, **kwargs)

    def redistribute(self, logits, index):
        return logits - torch.inf(device=logits) * self.eye(index)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        device = inputs.device
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        X = self.embed1(inputs.view(T * N, -1).long()).view(T, N, -1)
        actions = self.parse_actions(actions)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)
        new = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        new[0] = False  # TODO

        # search (needed for log_probs)
        values = hx.values.view(N, self.planning_steps, self.num_options)
        search_probs = hx.search_probs.view(values.shape)
        search_options = actions.searched.view(T, N, self.planning_steps).long()[0, new]

        # indexes
        I = torch.zeros_like(new[new]).long()
        R = torch.arange(I.numel(), device=device)

        # plan  (needed for execution)
        planned_h = torch.zeros(
            (new.sum(), self.planning_steps, self.hidden_size, self.num_model_layers),
            device=device,
        )
        planned_options = hx.planned_options.view(N, self.planning_steps).long()
        planned_states = torch.zeros(
            (new.sum(), self.planning_steps, self.embedding_size), device=device
        )
        planned_logits = torch.zeros_like(values)[new]
        planned_states[R, 0] = self.embed2(X[0, new])

        if new.any():
            for i in range(self.planning_steps):
                # TODO: prevent cycles?
                new_logits = (planned_logits[R, I] == 0).all(-1, keepdim=True)
                # if new_logits.any():
                x = planned_states[R, I]
                sharpness = self.sharpener(x)
                v = self.critic(x)
                values[new, I] = v
                logits = torch.where(new_logits, sharpness * v, planned_logits[R, I])
                # else:
                #     logits = planned_logits[R, I]
                dist = FixedCategorical(logits=logits)
                self.sample_new(search_options[:, i], dist)
                P = search_options[:, i]
                search_probs[new, i] = dist.probs

                # stack stuff
                planned_logits[R, I] = logits - INF * self.eye(P)
                pop = v[R, P] <= 0
                push = v[R, P] > 0
                planned_options[new, I] = P
                if push.any():
                    options = self.embed_options(planned_options[new, I][push])
                    model_input = torch.cat(
                        [planned_states[R, I][push], options], dim=-1
                    )
                    h = planned_h[push, I[push]]
                    model_output, h = self.model(
                        model_input.unsqueeze(0), h.permute(2, 0, 1)
                    )
                    planned_states[push, I[push] + 1] = self.embed2(
                        model_output.squeeze(0)
                    )
                    planned_h[push, I[push] + 1] = h.permute(1, 2, 0)
                    I[push] = torch.min(I + 1, self.planning_steps * self.one)[push]
                I[pop] = torch.max(I - 1, self.zero)[pop]
                # TODO: somehow add early termination

        # TODO: add obs to recurrence
        h = hx.h.view(N, self.num_model_layers, -1).transpose(0, 1)
        X = torch.cat([X, hx.x.unsqueeze(0)], dim=0)
        new_search_options = search_options.float()
        search_options = hx.search_options
        search_options[new] = new_search_options

        for t in range(T):
            option = planned_options[:, t]
            model_input = torch.cat(
                [self.embed2(X[t - 1]), self.embed_options(option)], dim=-1
            ).unsqueeze(0)
            model_output, h = self.model(model_input, h)
            model_loss = F.mse_loss(
                model_output[0], X[t].detach(), reduction="none"
            ).mean(-1)
            # TODO: predict from start of episode

            # TODO add gate (c) so that obs is not compared every turn
            # logits = F.cosine_similarity(X[t], X)
            # dist = FixedCategorical(logits=self.sharpener(inputs[t]) * logits)
            # embed_loss = -dist.entropy()  # maximize distinctions among embedded
            # self.sample_new(
            #     planned_options[t], dist
            # )

            yield RecurrentState(
                values=values,
                planned_options=planned_options,
                search_options=search_options,
                search_probs=search_probs,
                a=planned_options[:, t],
                model_loss=model_loss,
                embed_loss=hx.embed_loss,  # TODO
                h=h.transpose(0, 1),
                x=X[t],
                v=values[:, t].gather(-1, planned_options[:, t].unsqueeze(1)),
            )


"""
Questions:
- prevent cycles
- early termination
- gate network
- model loss
- embed loss?
"""
