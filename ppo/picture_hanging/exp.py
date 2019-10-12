import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import NNBase, AgentValues

from ppo.distributions import DiagGaussian, Categorical, FixedNormal, FixedCategorical
from ppo.picture_hanging.env import Obs
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a b a_loc a_scale b_probs v h p")


class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, entropy_coef, recurrence):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        self.recurrent_module = recurrence

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        rm = self.recurrent_module
        hx = rm.parse_hidden(all_hxs)
        a_dist = FixedNormal(loc=hx.a_loc, scale=hx.a_scale)
        b_dist = FixedCategorical(probs=hx.b_probs)
        action_log_probs = a_dist.log_probs(hx.a) + b_dist.log_probs(hx.b)
        entropy = (a_dist.entropy() + b_dist.entropy()).mean()
        return AgentValues(
            value=hx.v,
            action=torch.cat([hx.a, hx.b], dim=-1),
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=None,
            rnn_hxs=last_hx,
            log=dict(entropy=entropy),
        )

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            y = torch.cat([x, action.float()], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    def get_value(self, inputs, rnn_hxs, masks):
        all_hxs, last_hx = self._forward_gru(
            inputs.view(inputs.size(0), -1), rnn_hxs, masks
        )
        return self.recurrent_module.parse_hidden(last_hx).v


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        activation,
        hidden_size,
        num_layers,
        debug,
        bidirectional,
    ):
        super().__init__()
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.gru = nn.GRU(1, hidden_size, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        layers = []
        for i in range(max(0, num_layers - 1)):
            layers += [init_(nn.Linear(hidden_size, hidden_size)), activation]

        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size * num_directions, hidden_size)),
            *layers,
            DiagGaussian(hidden_size, action_space.spaces["goal"].shape[0])
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size * num_directions, hidden_size)),
            *copy.deepcopy(layers),
            init_(nn.Linear(hidden_size, 1))
        )
        self.beta = nn.Sequential(
            init_(
                nn.Linear(
                    hidden_size * num_directions + self.obs_sections.obs, hidden_size
                )
            ),
            *copy.deepcopy(layers),
            Categorical(hidden_size, 2)
        )
        self.state_sizes = RecurrentState(
            a=1, b=1, a_loc=1, a_scale=1, b_probs=2, p=1, v=1, h=hidden_size
        )

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

    def print(self, *args, **kwargs):
        if self.debug:
            torch.set_printoptions(precision=2, sci_mode=False)
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        inputs = self.parse_inputs(inputs)
        M, Mn = self.gru(inputs.sizes[0].T.unsqueeze(-1))

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        P = hx.p.squeeze(1).long()
        R = torch.arange(P.size(0), device=P.device)
        A = actions.clone()[:, :, 0]
        B = actions.clone()[:, :, 1].long()

        for t in range(T):
            r = M[P, R]
            v = self.critic(r)
            a_dist = self.actor(r)
            b_dist = self.beta(torch.cat([inputs.obs[t], r], dim=-1))
            self.sample_new(A[t], a_dist)
            self.sample_new(B[t], b_dist)
            yield RecurrentState(
                a=A[t],
                b=B[t],
                a_loc=a_dist.loc,
                a_scale=a_dist.scale,
                b_probs=b_dist.probs,
                v=v,
                h=hx.h,
                p=(P + B[t]) % (M.size(0)),
            )
