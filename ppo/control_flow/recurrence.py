from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

from ppo.control_flow.env import Action
from ppo.control_flow.multi_step.transformer import TransformerModel
from ppo.distributions import Categorical, FixedCategorical
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a d h p v a_probs d_probs P")


def get_obs_sections(obs_spaces):
    return [int(np.prod(s.shape)) for s in obs_spaces]


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        eval_lines,
        activation,
        hidden_size,
        gate_hidden_size,
        task_embed_size,
        num_layers,
        num_edges,
        num_encoding_layers,
        debug,
        no_scan,
        no_roll,
        no_pointer,
        transformer,
        olsk,
        log_dir,
    ):
        super().__init__()
        if olsk:
            num_edges = 3
        self.olsk = olsk
        self.no_pointer = no_pointer
        self.transformer = transformer
        self.log_dir = log_dir
        self.no_roll = no_roll
        self.no_scan = no_scan
        self.obs_spaces = observation_space
        self.action_size = action_space.nvec.size
        self.debug = debug
        self.hidden_size = hidden_size
        self.task_embed_size = task_embed_size

        self.obs_sections = self.get_obs_sections(self.obs_spaces)
        self.eval_lines = eval_lines
        self.train_lines = len(self.obs_spaces.lines.nvec)

        # networks
        self.ne = num_edges
        self.action_space_nvec = Action(*map(int, action_space.nvec))
        n_a = self.action_space_nvec.upper
        self.n_a = n_a
        self.embed_task = self.build_embed_task(task_embed_size)
        self.embed_upper = nn.Embedding(n_a, hidden_size)
        self.task_encoder = (
            TransformerModel(
                ntoken=self.ne * self.d_space(),
                ninp=task_embed_size,
                nhid=task_embed_size,
            )
            if transformer
            else nn.GRU(
                task_embed_size, task_embed_size, bidirectional=True, batch_first=True
            )
        )
        # self.minimal_gru.py = nn.GRUCell(self.gru_in_size, gru_hidden_size)

        # layers = []
        # in_size = gru_hidden_size + 1
        # for _ in range(num_layers):
        # layers.extend([init_(nn.Linear(in_size, hidden_size)), activation])
        # in_size = hidden_size
        # self.zeta2 = nn.Sequential(*layers)
        if self.olsk:
            assert self.ne == 3
            self.upsilon = nn.GRUCell(gate_hidden_size, hidden_size)
            self.beta = init_(nn.Linear(hidden_size, self.ne))
        elif self.no_pointer:
            self.upsilon = nn.GRUCell(gate_hidden_size, hidden_size)
            self.beta = init_(nn.Linear(hidden_size, self.d_space()))
        else:
            self.upsilon = init_(nn.Linear(gate_hidden_size, self.ne))
            layers = []
            in_size = (2 if self.no_roll or self.no_scan else 1) * task_embed_size
            for _ in range(num_encoding_layers - 1):
                layers.extend([init_(nn.Linear(in_size, task_embed_size)), activation])
                in_size = task_embed_size
            out_size = self.ne * self.d_space() if self.no_scan else self.ne
            self.beta = nn.Sequential(*layers, init_(nn.Linear(in_size, out_size)))
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, n_a)
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=n_a,
            d=1,
            d_probs=(self.d_space()),
            h=hidden_size,
            p=1,
            v=1,
            P=(self.P_shape().prod()),
        )

    def P_shape(self):
        lines = (
            self.obs_spaces["lines"]
            if isinstance(self.obs_spaces, dict)
            else self.obs_spaces.lines
        )
        if self.olsk or self.no_pointer:
            return np.zeros(1, dtype=int)
        else:
            return np.array([len(lines.nvec), self.d_space(), self.ne])

    def d_space(self):
        if self.olsk:
            return 3
        elif self.transformer or self.no_scan or self.no_pointer:
            return 2 * self.eval_lines
        else:
            return 2 * self.train_lines

    def build_embed_task(self, hidden_size):
        return nn.Embedding(self.obs_spaces.lines.nvec[0], hidden_size)

    @property
    def gru_in_size(self):
        return self.task_embed_size + self.ne

    @staticmethod
    def get_obs_sections(obs_spaces):
        return get_obs_sections(obs_spaces)

    # noinspection PyProtectedMember
    @contextmanager
    def evaluating(self, eval_obs_space):
        obs_spaces = self.obs_spaces
        obs_sections = self.obs_sections
        state_sizes = self.state_sizes
        train_lines = self.train_lines
        self.set_obs_space(eval_obs_space)
        yield self
        self.obs_spaces = obs_spaces
        self.obs_sections = obs_sections
        self.state_sizes = state_sizes
        self.train_lines = train_lines

    def set_obs_space(self, obs_space):
        self.obs_spaces = obs_space.spaces
        self.obs_sections = self.get_obs_sections(self.obs_spaces)
        self.train_lines = len(self.obs_spaces["lines"].nvec)
        # noinspection PyProtectedMember
        self.state_sizes = self.state_sizes._replace(
            d_probs=self.d_space(), P=self.P_shape().prod()
        )

    @staticmethod
    def get_lines_space(n_eval_lines, train_lines_space):
        return spaces.MultiDiscrete(
            np.repeat(train_lines_space.nvec[:1], repeats=n_eval_lines, axis=0)
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

    def parse_obs(self, inputs: torch.Tensor):
        return torch.split(inputs, self.obs_sections, dim=-1)

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        args = [
            torch.round(100 * a)
            if type(a) is torch.Tensor and a.dtype == torch.float
            else a
            for a in args
        ]
        if self.debug:
            print(*args, **kwargs)

    def build_P(self, M, N, device, nl):
        if self.no_pointer:
            _, G = self.task_encoder(M)
            return G.transpose(0, 1).reshape(N, -1)
        if self.no_scan:
            if self.no_roll:
                H, _ = self.task_encoder(M)
            else:
                rolled = torch.cat(
                    [torch.roll(M, shifts=-i, dims=1) for i in range(nl)], dim=0
                )
                _, H = self.task_encoder(rolled)
            H = H.transpose(0, 1).reshape(nl, N, -1)
            P = self.beta(H).view(nl, N, -1, self.ne).softmax(2)
            return P
        elif self.transformer:
            P = self.task_encoder(M.transpose(0, 1)).view(nl, N, -1, self.ne).softmax(2)
            return P
        else:
            if self.no_roll:
                G, _ = self.task_encoder(M)
                G = torch.cat(
                    [
                        G.unsqueeze(1).expand(-1, nl, -1, -1),
                        G.unsqueeze(2).expand(-1, -1, nl, -1),
                    ],
                    dim=-1,
                ).transpose(0, 1)
            else:
                rolled = torch.cat(
                    [torch.roll(M, shifts=-i, dims=1) for i in range(nl)], dim=0
                )
                G, _ = self.task_encoder(rolled)
            G = G.view(nl, N, nl, 2, -1)
            B = bb = self.beta(G).sigmoid()
            # arange = torch.zeros(6).float()
            # arange[0] = 1
            # arange[1] = 1
            # B[:, :, :, 0] = 0  # arange.view(1, 1, -1, 1)
            # B[:, :, :, 1] = 1
            f, b = torch.unbind(B, dim=3)
            B = torch.stack([f, b.flip(2)], dim=-2)
            B = B.view(nl, N, 2 * nl, self.ne)
            last = torch.zeros(nl, N, 2 * nl, self.ne, device=device)
            last[:, :, -1] = 1
            B = (1 - last).flip(2) * B  # this ensures the first B is 0
            zero_last = (1 - last) * B
            B = zero_last + last  # this ensures that the last B is 1
            rolled = torch.roll(zero_last, shifts=1, dims=2)
            C = torch.cumprod(1 - rolled, dim=2)
            P = B * C
            P = P.view(nl, N, nl, 2, self.ne)
            f, b = torch.unbind(P, dim=3)
            P = torch.cat([b.flip(2), f], dim=2)
            return P

    def build_memory(self, N, T, inputs):
        lines = inputs.lines.view(T, N, self.obs_sections.lines).long()[0]
        return self.embed_task(lines.view(-1)).view(
            *lines.shape, self.task_embed_size
        )  # n_batch, n_lines, hidden_size

    @staticmethod
    def preprocess_obs(obs):
        return obs.unsqueeze(-1)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, dim = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_obs(inputs)

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        M = self.build_memory(N, T, inputs)

        P = self.build_P(M, N, rnn_hxs.device, nl)
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        p = hx.p.long().squeeze(-1)
        a = hx.a.long().squeeze(-1)
        a[new_episode] = 0
        u = hx.u
        R = torch.arange(N, device=rnn_hxs.device)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            obs = inputs.obs[t]
            h = self.gru(torch.cat([M[R, p], u], dim=-1), h)
            z = F.relu(self.zeta2(torch.cat([obs, h], dim=-1)))
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)
            u = self.upsilon(z).softmax(dim=-1)
            self.print("u", u)
            w = P[p, R]
            half1 = w.size(1) // 2
            self.print(w[0, half1:])
            self.print(w[0, :half1])
            d_dist = FixedCategorical(probs=((w @ u.unsqueeze(-1)).squeeze(-1)))
            # p_probs = torch.round(p_dist.probs * 10).flatten()
            self.sample_new(D[t], d_dist)
            n_p = d_dist.probs.size(-1)
            p = p + D[t].clone() - n_p // 2
            p = torch.clamp(p, min=0, max=M.size(1) - 1)
            d = D[t]
            yield RecurrentState(
                a=A[t],
                h=h,
                v=self.critic(z),
                p=p,
                a_probs=a_dist.probs,
                d=d,
                d_probs=d_dist.probs,
            )
