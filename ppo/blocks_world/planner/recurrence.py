from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.blocks_world.planner.env import Obs
from ppo.distributions import FixedCategorical
from ppo.layers import Flatten
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState", "a p v index actions probs states model_loss"
)


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
        always_plan,
    ):
        self.always_plan = always_plan
        self.input_sections = Obs(
            *[int(np.prod(s.shape)) for s in observation_space.spaces.values()]
        )
        num_inputs = self.input_sections.obs
        no = observation_space.spaces["obs"].nvec.max()
        na = action_space.n
        super().__init__()
        self.action_size = 1
        self.planning_steps = planning_steps

        self.state_sizes = RecurrentState(
            a=1,
            p=na,
            index=1,
            v=1,
            actions=planning_steps,
            probs=planning_steps * na,
            states=planning_steps * embedding_size,
            model_loss=1,
        )

        # networks
        self.embed_action = nn.Embedding(na, na)
        layers = [nn.Embedding(no, no), Flatten()]
        in_size = no * num_inputs
        for _ in range(num_embedding_layers):
            layers += [activation, init_(nn.Linear(in_size, hidden_size))]
            in_size = hidden_size
        self.embed1 = nn.Sequential(*layers)
        self.embed2 = nn.Sequential(
            activation, init_(nn.Linear(hidden_size, embedding_size))
        )
        self.model = nn.GRU(
            int(embedding_size + self.embed_action.embedding_dim),
            hidden_size,
            num_model_layers,
        )

        self.critic = init_(nn.Linear(embedding_size, 1))
        self.actor = init_(nn.Linear(embedding_size, na))
        self.train()

    def print(self, t, *args, **kwargs):
        if self.debug:
            torch.set_printoptions(precision=2)
            print(t, *args, **kwargs)

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, rnn_hxs):
        return self.pack(self.inner_loop(inputs, rnn_hxs))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def parse_inputs(self, inputs: torch.Tensor) -> list:
        return Obs(*torch.split(inputs, self.input_sections, dim=-1))

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
        inputs, input_actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        inputs = self.parse_inputs(inputs).obs.long()

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        new = torch.all(rnn_hxs == 0, dim=-1)
        if new.any():
            assert new.all()
        if new.any() or self.always_plan:
            h = (
                torch.zeros(
                    N,
                    self.model.num_layers,
                    self.model.hidden_size,
                    device=rnn_hxs.device,
                )
                .transpose(0, 1)
                .contiguous()
            )

            state = self.embed2(self.embed1(inputs[0]))
            new_actions = []
            states = []
            probs = []
            for i in range(self.planning_steps):
                states.append(state)
                dist = FixedCategorical(logits=self.actor(state))
                new_actions.append(dist.sample())
                probs.append(dist.probs)
                model_input = torch.cat(
                    [state, self.embed_action(new_actions[-1].squeeze(1).clone())],
                    dim=-1,
                )
                hn, h = self.model(model_input.unsqueeze(0), h)
                state = self.embed2(hn.squeeze(0))
            recurrent_actions = torch.stack(new_actions, dim=1).float()
            probs = torch.stack(probs, dim=1)
            states = torch.stack(states, dim=1)
        else:
            recurrent_actions = hx.actions.view(N, self.planning_steps, 1)
            probs = hx.probs.view(N, self.planning_steps, -1)
            states = hx.states.view(N, self.planning_steps, -1)

        index = hx.index
        for t in range(T):
            i = int(torch.mean(index))
            x = self.embed2(self.embed1(inputs[t]))
            v = self.critic(x)
            model_loss = F.mse_loss(states[:, t], x.detach(), reduction="none").mean(1)
            a = input_actions[t].where(input_actions[t] >= 0, recurrent_actions[:, i])
            index += 1
            yield RecurrentState(
                a=a,
                p=probs[:, i],
                index=index,
                v=v,
                actions=recurrent_actions,
                probs=probs,
                states=states,
                model_loss=model_loss,
            )
