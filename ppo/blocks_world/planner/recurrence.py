from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from ppo.distributions import Categorical, FixedCategorical
from ppo.layers import Flatten
from ppo.mdp.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a probs planned_probs plan v t ")
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
        self.action_size = 1 + planning_steps
        self.debug = debug
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        nvec = observation_space.nvec
        self.obs_shape = (*nvec.shape, nvec.max())
        self.num_options = nvec.max()
        self.hidden_size = hidden_size

        self.state_sizes = RecurrentState(
            a=planning_steps,
            plan=planning_steps,
            v=1,
            t=1,
            probs=planning_steps * action_space.nvec.max(),
            planned_probs=planning_steps * action_space.nvec.max(),
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
        self.embed_action = nn.Embedding(
            int(action_space.nvec.max()), int(action_space.nvec.max())
        )

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

        self.actor = Categorical(embedding_size, action_space.nvec.max())
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

        plan = hx.plan
        planned_probs = hx.planned_probs.view(N, self.planning_steps, -1)

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

            _, *plan = torch.split(actions[0].long(), 1, dim=-1)
            state = self.embed2(self.embed1(inputs[0]))
            probs = []
            for t in range(self.planning_steps):
                dist = self.actor(state)  # page 7 left column
                probs.append(dist.probs)
                self.sample_new(plan[t], dist)
                model_input = torch.cat(
                    [state, self.embed_action(plan[t].squeeze(1))], dim=-1
                ).unsqueeze(0)
                hn, h = self.model(model_input, h)
                state = self.embed2(hn.squeeze(0))

            planned_probs = torch.stack(probs, dim=1)
            plan = torch.cat(plan, dim=-1)

        for t in range(T):
            value = self.critic(self.embed2(self.embed1(inputs[t])))
            t = hx.t[0].long().item()
            yield RecurrentState(
                a=plan,
                planned_probs=planned_probs,
                plan=plan,
                probs=planned_probs,
                v=value,
                t=hx.t + 1,
            )
