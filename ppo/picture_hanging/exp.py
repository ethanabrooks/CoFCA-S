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

RecurrentState = namedtuple("RecurrentState", "a b probs v h p")


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
        a_dist = FixedCategorical(probs=hx.probs)
        # b_dist = FixedCategorical(probs=hx.b_probs)
        action_log_probs = a_dist.log_probs(hx.a)  # + b_dist.log_probs(hx.b)
        entropy = (a_dist.entropy()).mean()
        return AgentValues(
            value=hx.v,
            # action=torch.cat([hx.a, hx.b], dim=-1),
            action=hx.a,
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
        scale,
    ):
        super().__init__()
        self.default_scale = scale
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(sizes=self.obs_spaces.sizes.nvec.size, pos=1, index=1)
        self.action_size = 1
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.embed_action = nn.Embedding(action_space.n, hidden_size)
        self.embed_pos = nn.Embedding(observation_space.spaces["pos"].n, hidden_size)
        self.embed_index = nn.Embedding(
            observation_space.spaces["index"].n, hidden_size
        )
        self.gru = nn.GRU(1, hidden_size, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        layers = []
        for i in range(max(0, num_layers - 1)):
            layers += [init_(nn.Linear(hidden_size, hidden_size)), activation]
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size * num_directions, hidden_size)),
            *layers,
            init_(nn.Linear(hidden_size, action_space.n - 1)),
        )
        # self.scale = nn.Sequential(
        #     init_(nn.Linear(hidden_size, hidden_size)),
        #     *layers,
        #     init_(nn.Linear(hidden_size, 1)),
        #     nn.Softplus()
        # )
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            *copy.deepcopy(layers),
            init_(nn.Linear(hidden_size, 1)),
        )
        self.beta = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            *copy.deepcopy(layers),
            init_(nn.Linear(hidden_size, 1)),
        )
        self.controller = nn.GRUCell(hidden_size * (3 + num_directions), hidden_size)
        self.register_buffer("next", torch.eye(action_space.n)[-1])
        self.state_sizes = RecurrentState(
            a=1,
            b=1,
            # a_loc=1,
            # a_scale=1,
            # b_probs=2,
            probs=action_space.n,
            p=1,
            v=1,
            h=hidden_size,
            # n=1,
            # right=1,
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
        # device = inputs.device
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )
        inputs = self.parse_inputs(inputs)
        M, Mn = self.gru(inputs.sizes[0].T.unsqueeze(-1))

        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        # n = hx.n.long().squeeze(-1)
        # right = hx.right.squeeze(-1)
        # I = torch.arange(N, device=device)

        index = inputs.index.long().squeeze(-1)
        pos = inputs.pos.long().squeeze(-1)
        h = hx.h
        P = hx.p.squeeze(1).long()
        R = torch.arange(P.size(0), device=P.device)
        A = torch.cat([actions.clone()[:, :, 0], hx.a.T], dim=0).long()
        # B = actions.clone()[:, :, 1].long()

        for t in range(T):
            # a = A[t - 1]
            r = M[P, R]
            x = torch.cat(
                [
                    self.embed_index(index[t]),
                    self.embed_pos(pos[t]),
                    self.embed_action(A[t - 1].clone()),
                    r,
                ],
                dim=-1,
            )
            y = self.controller(x, h)
            b = self.beta(y).sigmoid()
            # self.sample_new(B[t], b_dist)
            # b = B[t].float().unsqueeze(-1)
            v = self.critic(y)

            # picture_size = inputs.sizes[t, I, n]
            # a = right + picture_size / 2
            # P = (P + B[t]) % (M.size(0))
            # n = (n + B[t]) % (M.size(0))
            # right = right + picture_size * B[t].float()
            a_probs = self.actor(r).softmax(-1)
            dist = FixedCategorical((1 - b) * F.pad(a_probs, [0, 1]) + b * self.next)

            # a_dist = FixedNormal(
            #     loc=b * a_dist.loc + (1 - b) * hx.a,
            #     scale=b * a_dist.scale + (1 - b) * self.scale(y),
            # )
            self.sample_new(A[t], dist)
            b = (A[t] == a_probs.size(1)).long()
            yield RecurrentState(
                a=A[t],
                # b=B[t],
                b=b,
                probs=dist.probs,
                # a_loc=a_dist.loc,
                # a_scale=a_dist.scale,
                # b_probs=b_dist.probs,
                v=v,
                h=h,
                p=(P + b) % M.size(0),
                # right=right,
            )
