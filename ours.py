from contextlib import contextmanager
from dataclasses import replace, astuple, dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from agents import MultiEmbeddingBag
from data_types import (
    Obs,
    RecurrentState,
    ParsedInput,
    Action,
)
from distribution_modules import FixedCategorical, Categorical
from lower_agent import get_obs_sections, optimal_padding
from transformer import TransformerModel
from env import ActionSpace
from utils import init_, init


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


@dataclass
class Recurrence(nn.Module):
    action_space: ActionSpace
    conv_hidden_size: int
    debug: bool
    debug_obs: bool
    gate_coef: float
    hidden_size: int
    inventory_hidden_size: int
    kernel_size: int
    lower_embed_size: int
    max_eval_lines: int
    no_pointer: bool
    no_roll: bool
    no_scan: bool
    num_edges: int
    observation_space: spaces.Dict
    olsk: bool
    stride: int
    task_embed_size: int
    transformer: bool

    def __post_init__(self):
        super().__init__()
        obs_spaces = Obs(**self.observation_space.spaces)
        self.obs_sections = get_obs_sections(obs_spaces)
        self.train_lines = len(self.obs_spaces.lines.nvec)

        # networks
        if self.olsk:
            num_edges = 3
        self.embed_task = MultiEmbeddingBag(
            self.obs_spaces.lines.nvec[0], embedding_dim=self.task_embed_size
        )
        self.task_encoder = (
            TransformerModel(
                ntoken=self.ne * self.d_space(),
                ninp=self.task_embed_size,
                nhid=self.task_embed_size,
            )
            if self.transformer
            else nn.GRU(
                self.task_embed_size,
                self.task_embed_size,
                bidirectional=True,
                batch_first=True,
            )
        )

        action_nvec = Action(*self.action_space.nvec)
        self.actions_size = len(astuple(action_nvec))
        self.actor_input_size = (
            self.hidden_size + self.lower_embed_size
        )  # TODO: try multiplication
        self.actor = init(
            nn.Linear(
                self.actor_input_size, self.actions_size * self.action_space.nvec.max()
            ),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01,
        )  # TODO: try init_
        actor_nvec = torch.tensor(astuple(action_nvec.a_actions()))
        self.embed_lower = MultiEmbeddingBag(actor_nvec)
        self.conv_hidden_size = self.conv_hidden_size

        self.register_buffer("AR", torch.arange(len(actor_nvec)))
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))
        self.register_buffer(
            "thresholds", torch.tensor(astuple(action_nvec.thresholds()))
        )

        masks = torch.zeros(self.actions_size, self.action_space.nvec.max())
        masks[
            torch.arange(self.actions_size).unsqueeze(0) < actor_nvec.unsqueeze(1)
        ] = 1
        self.register_buffer("masks", masks)

        d, h, w = (2, 1, 1) if self.debug_obs else self.obs_spaces.obs.shape
        self.obs_dim = d
        self.kernel_size = min(d, self.kernel_size)
        self.padding = optimal_padding(h, self.kernel_size, self.stride)
        self.embed_inventory = nn.Sequential(
            init_(nn.Linear(self.obs_spaces.resources.n, self.inventory_hidden_size)),
            nn.ReLU(),
        )
        m_size = (
            2 * self.task_embed_size + self.hidden_size
            if self.no_pointer
            else self.task_embed_size
        )
        z_size = m_size + self.conv_hidden_size + self.inventory_hidden_size
        self.zeta1 = init_(nn.Linear(z_size, self.hidden_size))
        if self.olsk:
            assert self.ne == 3
            self.upsilon = nn.GRUCell(z_size, self.hidden_size)
            self.beta = init_(nn.Linear(self.hidden_size, self.ne))
        elif self.no_pointer:
            self.upsilon = nn.GRUCell(z_size, self.hidden_size)
            self.beta = init_(nn.Linear(self.hidden_size, self.d_space()))
        else:
            self.upsilon = init_(nn.Linear(z_size, self.ne))
            in_size = (2 if self.no_roll or self.no_scan else 1) * self.task_embed_size
            out_size = self.ne * self.d_space() if self.no_scan else self.ne
            self.beta = nn.Sequential(init_(nn.Linear(in_size, out_size)))
        self.d_gate = Categorical(z_size, 2)
        self.kernel_net = nn.Linear(
            m_size, self.conv_hidden_size * self.kernel_size ** 2 * d
        )
        self.conv_bias = nn.Parameter(torch.zeros(self.conv_hidden_size))
        self.critic = init_(nn.Linear(self.hidden_size, 1))
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=actor_nvec.max(),
            d=1,
            d_probs=(self.d_space()),
            h=self.hidden_size,
            p=1,
            v=1,
            l=1,
            dg_probs=2,
            dg=1,
        )

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        state_sizes = self.state_sizes
        # noinspection PyArgumentList
        if hx.size(-1) == sum(self.state_sizes):
            state_sizes = self.state_sizes
        return RecurrentState(*torch.split(hx, state_sizes, dim=-1))

    # noinspection PyPep8Naming
    def inner_loop(self, raw_inputs, rnn_hxs):
        T, N, dim = raw_inputs.shape
        nl = len(self.obs_spaces.lines.nvec)
        inputs = ParsedInput(
            *torch.split(
                raw_inputs,
                ParsedInput(obs=sum(self.obs_sections), actions=self.actions_size),
                dim=-1,
            )
        )

        # parse non-action inputs
        state = Obs(*torch.split(inputs.obs, self.obs_sections, dim=-1))
        state = replace(state, obs=state.obs.view(T, N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(T, N, *self.obs_spaces.lines.shape)[0].long()
        mask = state.mask[0].view(N, nl)
        mask = F.pad(mask, [0, nl])  # pad for backward mask
        mask = torch.stack(
            [torch.roll(mask, shifts=-i, dims=1) for i in range(nl)], dim=0
        )
        mask[:, :, 0] = 0  # prevent self-loops
        mask = mask.view(nl, N, 2, nl).transpose(2, 3).unsqueeze(-1)

        # build memory
        M = self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            N, -1, self.task_embed_size
        )
        rolled = torch.stack(
            [torch.roll(M, shifts=-i, dims=1) for i in range(nl)], dim=0
        )
        first = torch.zeros(2 * nl, device=raw_inputs.device)
        first[0] = 0.1
        first = first.view(1, -1, 1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        hx = RecurrentState(*[x.squeeze(0) for x in astuple(hx)])

        p = hx.p.long().squeeze(-1)
        h = hx.h
        hx.a[new_episode] = self.n_a - 1
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)

        actions = Action(*inputs.actions.unbind(dim=2))
        A = torch.cat(astuple(actions.a_actions()), dim=2)
        D = actions.delta.long()
        DG = actions.dg.long()

        for t in range(T):

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
            elif self.transformer:
                P = (
                    self.task_encoder(M.transpose(0, 1))
                    .view(nl, N, -1, self.ne)
                    .softmax(2)
                )
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
                    G, _ = self.task_encoder(rolled[p, R])
                G = G.view(N, nl, 2, -1)
                B = self.beta(G).sigmoid()
                B = B * mask[p, R]
                f, b = torch.unbind(B, dim=2)
                B = torch.stack([f, b.flip(1)], dim=2)
                B = B.view(N, 2 * nl, self.ne)
                rolledB = torch.roll(B, shifts=1, dims=1)
                assert torch.all(rolledB[:, 0] == 0)
                C = torch.cumprod(1 - rolledB, dim=1)
                B = (1 - first) * B + first  # small prob stop on first
                P = B * C
                P = P.view(N, nl, 2, self.ne)
                f, b = torch.unbind(P, dim=2)
                P = torch.cat([b.flip(1), f], dim=1)
                # noinspection PyArgumentList
            # noinspection PyArgumentList
            half = P.size(1) // 2 if self.no_scan else nl

            self.print("P", P)
            m = torch.cat([P, h], dim=-1) if self.no_pointer else M[R, p]
            conv_kernel = self.kernel_net(m).view(
                N,
                self.conv_hidden_size,
                self.obs_dim,
                self.kernel_size,
                self.kernel_size,
            )

            obs = state.obs[t]
            h1 = torch.cat(
                [
                    F.conv2d(
                        input=o.unsqueeze(0),
                        weight=k,
                        bias=self.conv_bias,
                        stride=self.stride,
                        padding=self.padding,
                    )
                    for o, k in zip(obs.unbind(0), conv_kernel.unbind(0))
                ],
                dim=0,
            ).relu()
            h1 = h1.sum(-1).sum(-1)
            inventory = state.resources[t]
            embedded_inventory = self.embed_inventory(inventory)
            z = torch.cat([m, h1, embedded_inventory], dim=-1)
            z1 = F.relu(self.zeta1(z))

            # noinspection PyTypeChecker
            above_threshold: torch.Tensor = (
                A[t - 1] > self.thresholds
            )  # meets condition to progress to next action
            sampled = A[t - 1] >= 0  # sampled on a previous time step
            above_threshold[~sampled] = True  # ignore unsampled
            # assert torch.all(sampled.sum(-1) == l + 1)
            above_thresholds = above_threshold.prod(-1)  # met all thresholds
            next_l = sampled.sum(-1) % D  # next l if all thresholds are met
            l: torch.Tensor = above_thresholds * next_l  # otherwise go back to 0
            copy = self.AR.unsqueeze(0) < l.unsqueeze(1)
            # actions accumulated from prev time steps
            A[t][copy] = A[t - 1][copy]  # copy accumulated actions from A[t-1]

            lower = A[t - 1, R, hx.l]  # previous lower
            embedded_lower = self.embed_lower(lower)
            a_logits = self.actor(torch.cat([z1, embedded_lower], dim=-1))
            a_probs = F.softmax(a_logits, dim=-1)
            a_dist = FixedCategorical(probs=a_probs * self.masks[hx.l])
            self.sample_new(A[t, R, l], a_dist)

            # self.print("a_probs", a_dist.probs)

            # while True:
            #    try:
            #        A[:] = float(input("A:"))
            #        assert torch.all(A < self.n_a)
            #        break
            #    except (ValueError, AssertionError):
            #        pass

            d_gate = self.d_gate(z)
            self.sample_new(DG[t], d_gate)
            dg = DG[t].unsqueeze(-1).float()

            if self.olsk or self.no_pointer:
                h = self.upsilon(z, h)
                u = self.beta(h).softmax(dim=-1)
                d_dist = gate(dg, u, ones)
                self.sample_new(D[t], d_dist)
                delta = D[t].clone() - 1
            else:
                u = self.upsilon(z).softmax(dim=-1)
                self.print("u", u)
                d_probs = (P @ u.unsqueeze(-1)).squeeze(-1)

                self.print("dg prob, dg", d_gate.probs[:, 1], dg)
                d_dist = gate(dg, d_probs, ones * half)
                self.print("d_probs", d_dist.probs[:, :half])
                self.print("d_probs", d_dist.probs[:, half:])
                self.sample_new(D[t], d_dist)
                # D[:] = float(input("D:")) + half
                delta = D[t].clone() - half
                self.print("D[t], delta", D[t], delta)
            p = p + delta
            p = torch.clamp(p, min=0, max=M.size(1) - 1)
            yield RecurrentState(
                a=A[t],
                v=self.critic(z1),
                h=h,
                p=p,
                d=D[t],
                l=l,
                dg=dg,
                a_probs=a_probs,
                d_probs=d_dist.probs,
                dg_probs=d_gate.probs,
            )

    @property
    def gru_in_size(self):
        return self.hidden_size + self.conv_hidden_size + self.encoder_hidden_size

    def d_space(self):
        if self.olsk:
            return 3
        elif self.transformer or self.no_scan or self.no_pointer:
            return 2 * self.eval_lines
        else:
            return 2 * self.train_lines

    # noinspection PyAttributeOutsideInit
    @contextmanager
    def evaluating(self, eval_obs_space):
        obs_spaces = self.obs_spaces
        obs_sections = self.obs_sections
        state_sizes = self.state_sizes
        train_lines = self.train_lines
        self.obs_spaces = eval_obs_space.spaces
        self.obs_sections = get_obs_sections(Obs(**self.obs_spaces))
        self.train_lines = len(self.obs_spaces["lines"].nvec)
        self.state_sizes = replace(self.state_sizes, d_probs=self.d_space())
        self.obs_spaces = Obs(**self.obs_spaces)
        yield self
        self.obs_spaces = obs_spaces
        self.obs_sections = obs_sections
        self.state_sizes = state_sizes
        self.train_lines = train_lines

    @staticmethod
    def get_lines_space(n_eval_lines, train_lines_space):
        return spaces.MultiDiscrete(
            np.repeat(train_lines_space.nvec[:1], repeats=n_eval_lines, axis=0)
        )

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, rnn_hxs):
        hxs = self.inner_loop(inputs, rnn_hxs)

        def pack():
            for size, (name, hx) in zip(
                self.state_sizes, asdict(RecurrentState(*zip(*hxs)))
            ):
                x = torch.stack(hx).float()
                assert np.prod(x.shape[2:]) == size
                yield x.view(*x.shape[:2], -1)

        rnn_hxs = torch.cat(list(pack()), dim=-1)
        return rnn_hxs, rnn_hxs[-1:]

    def print(self, *args, **kwargs):
        args = [
            torch.round(100 * a)
            if type(a) is torch.Tensor and a.dtype == torch.float
            else a
            for a in args
        ]
        if self.debug:
            print(*args, **kwargs)
