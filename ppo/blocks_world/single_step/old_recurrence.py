from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import Categorical, FixedCategorical
from ppo.layers import Flatten
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState", "a probs planned_probs plan v t state h model_loss"
)
XiSections = namedtuple("XiSections", "Kr Br kw bw e v F_hat ga gw Pi")


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
            plan=planning_steps,
            v=1,
            t=1,
            probs=action_space.n,
            planned_probs=planning_steps * action_space.n,
            state=embedding_size,
            h=hidden_size * num_model_layers,
            model_loss=1,
        )

        # networks
        self.embed_action = nn.Embedding(int(action_space.n), int(action_space.n))
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
        self.actor = Categorical(embedding_size, action_space.n)
        self.critic = init_(nn.Linear(embedding_size, 1))

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

        plan = hx.plan
        planned_probs = hx.planned_probs.view(N, self.planning_steps, -1)

        new = torch.all(rnn_hxs == 0, dim=-1)
        # if new.any():
        #     assert new.all()
        #     state = self.embed2(self.embed1(inputs[0]))
        # else:
        #     state = hx.state.view(N, -1)

        # h = (
        #     hx.h.view(N, self.model.num_layers, self.model.hidden_size)
        #     .transpose(0, 1)
        #     .contiguous()
        # )

        A = actions.long()[:, :, 0]

        for t in range(T):
            x = self.embed2(self.embed1(inputs[t])).detach()
            # model_loss = F.mse_loss(state, x, reduction="none").sum(-1)
            dist = self.actor(x)
            value = self.critic(x)
            self.sample_new(A[t], dist)
            # model_input = torch.cat([state, self.embed_action(A[t].clone())], dim=-1)
            # hn, h = self.model(model_input.unsqueeze(0), h)
            # state = self.embed2(hn.squeeze(0))
            yield RecurrentState(
                a=A[t],
                plan=plan,
                planned_probs=planned_probs,
                probs=dist.probs,
                v=value,
                t=hx.t + 1,
                state=hx.state,
                h=hx.h,
                model_loss=hx.model_loss,
                # h=h.transpose(0, 1),
                # model_loss=model_loss,
            )
