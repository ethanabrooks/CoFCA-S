from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import Categorical
from ppo.layers import Flatten
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState", "a a_probs planned_a planned_a_probs v h log_probs entropy"
)
XiSections = namedtuple("XiSections", "Kr Br kw bw e v F_hat ga gw Pi")


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
        num_slots,
        slot_size,
        embedding_size,
        num_heads,
        planning_steps,
        num_model_layers,
        num_embedding_layers,
    ):
        super().__init__()
        self.planning_steps = planning_steps
        self.action_size = 1
        self.debug = debug
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())
        self.num_options = nvec.max()
        self.hidden_size = hidden_size

        self.state_sizes = RecurrentState(
            a=1,
            planned_a=planning_steps,
            v=1,
            h=num_layers * hidden_size,
            log_probs=1,
            entropy=1,
            a_probs=action_space.n,
            planned_a_probs=planning_steps * action_space.n,
        )
        self.xi_sections = XiSections(
            Kr=num_heads * slot_size,
            Br=num_heads,
            kw=slot_size,
            bw=1,
            e=slot_size,
            v=slot_size,
            F_hat=num_heads,
            ga=1,
            gw=1,
            Pi=3 * num_heads,
        )

        # networks
        assert num_layers > 0
        self.embeddings = nn.Embedding(int(nvec.max()), int(nvec.max()))
        self.embed_action = nn.Embedding(int(action_space.n), int(action_space.n))

        self.gru = nn.GRU(int(embedding_size), hidden_size, num_layers)
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
            embedding_size + self.embed_action.embedding_dim,
            hidden_size,
            num_model_layers,
        )

        self.Wxi = nn.Sequential(
            activation,
            init_(nn.Linear(num_layers * hidden_size, sum(self.xi_sections))),
        )
        self.actor = Categorical(embedding_size, action_space.n)
        self.critic = init_(nn.Linear(embedding_size, 1))

        self.register_buffer("mem_one_hots", torch.eye(num_slots))

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

    def print(self, t, *args, **kwargs):
        if self.debug:
            if type(t) == torch.Tensor:
                t = (t * 10.0).round() / 10.0
            print(t, *args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        device = inputs.device
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        inputs = inputs.long()

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        P = hx.planned_a
        a_probs = hx.planned_a_probs

        new = torch.all(rnn_hxs == 0, dim=-1)
        if new.any():
            assert new.all()
            h = (
                torch.zeros(
                    N, self.model.num_layers, self.model.hidden_size, device=device
                )
                .transpose(0, 1)
                .contiguous()
            )

            # P = actions.long().squeeze(-1)
            P = [
                torch.zeros(N, device=device).long() for _ in range(self.planning_steps)
            ]  # TODO
            state = self.embed2(self.embed1(inputs[0]))
            probs = []
            for t in range(self.planning_steps):
                dist = self.actor(state)  # page 7 left column
                probs.append(dist.probs)
                self.sample_new(P[t], dist)
                model_input = torch.cat(
                    [state, self.embed_action(P[t])], dim=-1
                ).unsqueeze(0)
                hn, h = self.model(model_input, h)
                state = self.embed2(hn.squeeze(0))

            a_probs = torch.stack(probs, dim=1)
            P = torch.stack(P, dim=-1)

        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long()

        h = (
            hx.h.view(N, self.gru.num_layers, self.gru.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )

        for t in range(T):
            x = self.embed2(self.embed1(inputs[t]))
            hn, h = self.gru(x.view(1, N, -1), h)
            hT = (
                h.transpose(0, 1).reshape(N, -1).contiguous()
            )  # switch layer and batch dims
            state = self.embed2(hn.squeeze(0))

            # act
            dist = self.actor(state)  # page 7 left column
            value = self.critic(state)
            self.sample_new(A[t], dist)
            yield RecurrentState(
                a=A[t],
                planned_a=P,
                planned_a_probs=a_probs,
                a_probs=dist.probs,
                v=value,
                h=hT,
                log_probs=torch.ones_like(dist.log_probs(A[t])),
                entropy=dist.entropy(),
            )
