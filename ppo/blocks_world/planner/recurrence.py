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
    "RecurrentState", "values probs options indices model_loss embed_loss h x a v"
)
Actions = namedtuple("Actions", "actual options")

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
            probs=planning_steps * self.num_options,
            options=planning_steps,
            indices=planning_steps,
            model_loss=1,
            embed_loss=1,
            h=num_model_layers * hidden_size,
            x=embedding_size,
            a=1,
            v=1,
        )
        self.action_sections = Actions(actual=1, options=planning_steps)

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
        inputs = self.embed1(inputs.view(T * N, -1).long()).view(T, N, -1)
        new_actions = (actions < 0).any()
        if new_actions:
            assert (actions < 0).all()
        actions = self.parse_actions(actions)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)
        I = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        options = hx.options.view(N, self.planning_steps).long()

        # search (needed for log_probs)
        values = hx.values.view(N, self.planning_steps, self.num_options)
        probs = hx.probs.view(values.shape)
        indices = hx.indices.view(N, self.planning_steps).long()

        if I.any():
            indices = torch.zeros_like(indices)
            options = actions.options.view(T, N, self.planning_steps).long()[0]
            assert I.all()
            # search (needed for log_probs)
            values = torch.zeros_like(values)
            options = -torch.ones_like(options)
            hidden_states = torch.zeros(
                (N, self.planning_steps, self.hidden_size, self.num_model_layers),
                device=device,
            )
            logits = torch.zeros_like(probs)
            # TODO: delete some of these?

            # plan  (needed for execution)
            new_state = self.embed2(inputs[0])
            states = torch.zeros(
                (N, self.planning_steps, self.embedding_size), device=device
            )
            J = torch.zeros_like(I).long()

            for j in range(self.planning_steps):
                states = states.index_copy(1, j * self.one, new_state.unsqueeze(1))
                indices[I, J] = j
                # TODO: prevent cycles?

                x = states[I, J]
                sharpness = self.sharpener(x)
                v = self.critic(x)
                values[:, j] = v
                new_logits = (logits[I, J] == 0).all(-1, keepdim=True)
                l = torch.where(new_logits, sharpness * v, logits[I, J])

                dist = FixedCategorical(logits=l)
                P = options[:, j]
                self.sample_new(P, dist)
                logits = logits.index_copy(
                    1, j * torch.ones_like(J), l.unsqueeze(1) - INF * self.eye(P)
                )

                # stack stuff
                pop = v[I, P] <= 0
                push = v[I, P] > 0
                options[I, J] = P
                if push.any():
                    embedded_options = self.embed_options(options[I, J])
                    model_input = torch.cat([states[I, J], embedded_options], dim=-1)
                    h = hidden_states[I, J]
                    model_output, h = self.model(
                        model_input.unsqueeze(0), h.permute(2, 0, 1)
                    )
                    hidden_states = hidden_states.index_copy(
                        1, j * self.one, h.permute(1, 2, 0).unsqueeze(1)
                    )
                    new_state = self.embed2(model_output[0]).where(
                        push.unsqueeze(-1), states[:, j]
                    )
                    J = torch.min(J + 1, self.planning_steps * self.one - 1).where(
                        push, J
                    )
                J = torch.max(J - 1, self.zero).where(pop, J)
                # TODO: somehow add early termination

            probs = logits.softmax(-1)

        # TODO: add obs to recurrence
        h = hx.h.view(N, self.hidden_size, self.num_model_layers).permute(2, 0, 1)
        inputs = torch.cat([self.embed2(inputs), hx.x.unsqueeze(0)], dim=0)
        I = torch.ones_like(I).bool()

        for t in range(T):
            J = indices[:, t]
            option = options[I, J]
            model_input = torch.cat(
                [inputs[t - 1], self.embed_options(option)], dim=-1
            ).unsqueeze(0)
            model_output, h = self.model(model_input, h)
            model_loss = F.mse_loss(
                self.embed2(model_output[0]), inputs[t].detach(), reduction="none"
            ).mean(-1)
            # TODO: predict from start of episode

            # TODO add gate (c) so that obs is not compared every turn

            yield RecurrentState(
                values=values,
                options=options,
                indices=indices,
                probs=probs,
                a=options[I, J],
                model_loss=model_loss,
                embed_loss=hx.embed_loss,  # TODO
                h=h.permute(1, 2, 0),
                x=inputs[t],
                v=values[I, J, option.long()],
            )


"""
Questions:
- prevent cycles
- early termination
- gate network
- model loss
- embed loss?
"""
