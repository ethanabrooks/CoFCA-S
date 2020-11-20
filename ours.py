from collections import Hashable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from agents import MultiEmbeddingBag
from data_types import ParsedInput, RecurrentState, RawAction
from distributions import FixedCategorical
from env import Obs
from transformer import TransformerModel
from utils import astuple, asdict, init


def optimal_padding(h, kernel, stride):
    n = np.ceil((h - kernel) / stride + 1)
    return int(np.ceil((stride * (n - 1) + kernel - h) / 2))


def conv_output_dimension(h, padding, kernel, stride, dilation=1):
    return int(1 + (h + 2 * padding - dilation * (kernel - 1) - 1) / stride)


def get_obs_sections(obs_spaces):
    return [int(np.prod(s.shape)) for s in obs_spaces]


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


@dataclass
class Recurrence(nn.Module):
    action_space: spaces.MultiDiscrete
    conv_hidden_size: int
    debug: bool
    hidden_size: int
    kernel_size: int
    lower_embed_size: int
    max_eval_lines: int
    no_pointer: bool
    no_roll: bool
    no_scan: bool
    num_edges: int
    observation_space: spaces.Dict
    olsk: bool
    resources_hidden_size: int
    stride: int
    task_embed_size: int
    transformer: bool

    def __hash__(self):
        return hash(tuple(x for x in astuple(self) if isinstance(x, Hashable)))

    def __post_init__(self):
        super().__init__()
        self.obs_spaces = Obs(**self.observation_space.spaces)
        self.action_size = self.action_space.nvec.size

        self.obs_sections = get_obs_sections(self.obs_spaces)
        self.eval_lines = self.max_eval_lines
        self.train_lines = len(self.obs_spaces.lines.nvec)

        action_nvec = RawAction(*map(int, self.action_space.nvec))

        self.embed_task = MultiEmbeddingBag(
            self.obs_spaces.lines.nvec[0], embedding_dim=self.task_embed_size
        )
        self.embed_lower = MultiEmbeddingBag(
            self.obs_spaces.partial_action.nvec, embedding_dim=self.lower_embed_size
        )
        self.task_encoder = (
            TransformerModel(
                ntoken=self.num_edges * self.d_space(),
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

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )  # TODO: try init
        self.actor = init_(
            nn.Linear(
                self.hidden_size,
                action_nvec.a,
            )
        )
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))

        d, h, w = self.obs_spaces.obs.shape
        d += 1
        self.obs_dim = d
        self.kernel_size = min(d, self.kernel_size)
        self.padding = optimal_padding(h, self.kernel_size, self.stride) + 1
        self.embed_inventory = nn.Sequential(
            init_(nn.Linear(self.obs_spaces.inventory.n, self.resources_hidden_size)),
            nn.ReLU(),
        )
        m_size = (
            2 * self.task_embed_size + self.hidden_size
            if self.no_pointer
            else self.task_embed_size
        )
        zeta1_input_size = (
            m_size
            + self.conv_hidden_size
            + self.resources_hidden_size
            + self.lower_embed_size
        )
        self.zeta1 = init_(nn.Linear(zeta1_input_size, self.hidden_size))
        if self.olsk:
            assert self.num_edges == 3
            self.upsilon = nn.GRUCell(zeta1_input_size, self.hidden_size)
            self.beta = init_(nn.Linear(self.hidden_size, self.num_edges))
        elif self.no_pointer:
            self.upsilon = nn.GRUCell(zeta1_input_size, self.hidden_size)
            self.beta = init_(nn.Linear(self.hidden_size, self.d_space()))
        else:
            self.upsilon = init_(nn.Linear(zeta1_input_size, self.num_edges))
            in_size = (2 if self.no_roll or self.no_scan else 1) * self.task_embed_size
            out_size = (
                self.num_edges * self.d_space() if self.no_scan else self.num_edges
            )
            self.beta = nn.Sequential(init_(nn.Linear(in_size, out_size)))
        self.d_gate = init_(nn.Linear(zeta1_input_size, 2))
        self.kernel_net = nn.Linear(
            m_size, self.conv_hidden_size * self.kernel_size ** 2 * d
        )
        self.conv_bias = nn.Parameter(torch.zeros(self.conv_hidden_size))
        self.critic = init_(nn.Linear(self.hidden_size, 1))
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_nvec.a,
            d=1,
            d_probs=(self.d_space()),
            h=self.hidden_size,
            p=1,
            v=1,
            dg_probs=2,
            dg=1,
        )

    def d_space(self):
        if self.olsk:
            return 3
        elif self.transformer or self.no_scan or self.no_pointer:
            return 2 * self.eval_lines
        else:
            return 2 * self.train_lines

    # PyAttributeOutsideInit
    @contextmanager
    def evaluating(self, eval_obs_space):
        obs_spaces = self.obs_spaces
        obs_sections = self.obs_sections
        state_sizes = self.state_sizes
        train_lines = self.train_lines
        self.obs_spaces = eval_obs_space.spaces
        self.obs_sections = get_obs_sections(Obs(**self.obs_spaces))
        self.train_lines = len(self.obs_spaces["lines"].nvec)
        # noinspection PyProtectedMember
        self.state_sizes = self.state_sizes._replace(d_probs=self.d_space())
        self.obs_spaces = Obs(**self.obs_spaces)
        yield self
        self.obs_spaces = obs_spaces
        self.obs_sections = obs_sections
        self.state_sizes = state_sizes
        self.train_lines = train_lines

    def forward(self, inputs, rnn_hxs):
        hxs = self.inner_loop(inputs, rnn_hxs)

        def pack():
            states = RecurrentState(*zip(*map(astuple, hxs)))
            for size, (name, hx) in zip(
                astuple(self.state_sizes),
                asdict(states).items(),
            ):
                x = torch.stack(hx).float()
                assert np.prod(x.shape[2:]) == size
                yield x.view(*x.shape[:2], -1)

        rnn_hxs = torch.cat(list(pack()), dim=-1)
        return rnn_hxs, rnn_hxs[-1:]

    @staticmethod
    def get_lines_space(n_eval_lines, train_lines_space):
        return spaces.MultiDiscrete(
            np.repeat(train_lines_space.nvec[:1], repeats=n_eval_lines, axis=0)
        )

    @property
    def gru_in_size(self):
        return self.hidden_size + self.conv_hidden_size + self.encoder_hidden_size

    # noinspection PyPep8Naming
    def inner_loop(self, raw_inputs, rnn_hxs):
        T, N, dim = raw_inputs.shape
        inputs = ParsedInput(
            *torch.split(
                raw_inputs,
                astuple(
                    ParsedInput(obs=sum(self.obs_sections), actions=self.action_size)
                ),
                dim=-1,
            )
        )

        # parse non-action inputs
        state = Obs(*torch.split(inputs.obs, self.obs_sections, dim=-1))
        state = state._replace(obs=state.obs.view(T, N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(T, N, *self.obs_spaces.lines.shape)[0].long()
        # mask = state.mask[0].view(N, nl)
        # mask = F.pad(mask, [0, nl], value=1)  # pad for backward mask
        # mask = torch.stack(
        #     [torch.roll(mask, shifts=-i, dims=1) for i in range(nl)], dim=0
        # )
        # mask[:, :, 0] = 0  # prevent self-loops
        # mask = mask.view(nl, N, 2, nl).transpose(2, 3).unsqueeze(-1)

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        M = self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            N, -1, self.task_embed_size
        )
        rolled = torch.stack(
            [torch.roll(M, shifts=-i, dims=1) for i in range(nl)], dim=0
        )
        # first = torch.zeros(2 * nl, device=raw_inputs.device)
        # first[0] = 0.1
        # first = first.view(1, -1, 1)
        last = torch.zeros(2 * nl, device=rnn_hxs.device)
        last[-1] = 1
        last = last.view(1, -1, 1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        hx = RecurrentState(*[_x.squeeze(0) for _x in astuple(hx)])

        p = hx.p.long().squeeze(-1)
        h = hx.h
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        actions = RawAction(*inputs.actions.unbind(dim=2))
        A = torch.cat([actions.a, hx.a.view(1, N)], dim=0).long().unsqueeze(-1)
        D = torch.cat([actions.delta], dim=0).long()
        DG = torch.cat([actions.dg], dim=0).long()

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
                P = self.beta(H).view(nl, N, -1, self.num_edges).softmax(2)
            elif self.transformer:
                P = (
                    self.task_encoder(M.transpose(0, 1))
                    .view(nl, N, -1, self.num_edges)
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
                # B = B * mask[p, R]
                f, b = torch.unbind(B, dim=-2)
                B = torch.stack([f, b.flip(-2)], dim=-2)
                B = B.view(N, 2 * nl, self.num_edges)
                B = (1 - last).flip(-2) * B  # this ensures the first B is 0
                zero_last = (1 - last) * B
                B = zero_last + last  # this ensures that the last B is 1
                C = torch.cumprod(1 - torch.roll(zero_last, shifts=1, dims=-2), dim=-2)
                P = B * C
                P = P.view(N, nl, 2, self.num_edges)
                f, b = torch.unbind(P, dim=-2)
                P = torch.cat([b.flip(-2), f], dim=-2)
                # noinspection PyArgumentList
                half = P.size(2) // 2 if self.no_scan else nl
            self.print("p", p)
            m = torch.cat([P, h], dim=-1) if self.no_pointer else M[R, p]
            conv_kernel = self.kernel_net(m).view(
                N,
                self.conv_hidden_size,
                self.obs_dim,
                self.kernel_size,
                self.kernel_size,
            )

            obs = (
                torch.stack(
                    [state.truthy[t][R, p], state.subtask_complete[t].squeeze(-1)],
                    dim=-1,
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

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
            inventory = self.embed_inventory(state.inventory[t])
            partial_action = self.embed_lower(state.partial_action[t].long())
            zeta1_input = torch.cat([m, h1, inventory, partial_action], dim=-1)
            z1 = F.relu(self.zeta1(zeta1_input))
            a_logits = self.actor(z1)
            a_probs = F.softmax(a_logits, dim=-1) * state.action_mask[t]
            a_dist = FixedCategorical(probs=a_probs)
            self.sample_new(A[t], a_dist)
            self.print("a_probs", a_dist.probs)

            d_logits = self.d_gate(zeta1_input)
            d_probs = F.softmax(d_logits, dim=-1)
            complete = state.action_complete[t].long()
            d_gate = gate(complete, d_probs, ones * 0)
            self.sample_new(DG[t], d_gate)
            dg = DG[t].unsqueeze(-1).float()

            if self.olsk or self.no_pointer:
                h = self.upsilon(zeta1_input, h)
                u = self.beta(h).softmax(dim=-1)
                d_dist = gate(dg, u, ones)
                self.sample_new(D[t], d_dist)
                delta = D[t].clone() - 1
            else:
                u = self.upsilon(zeta1_input).softmax(dim=-1)
                self.print("u", u)
                d_probs = (P @ u.unsqueeze(-1)).squeeze(-1)

                self.print("dg prob", d_gate.probs[:, 1])
                self.print("dg", dg)
                d_dist = gate(dg, d_probs, ones * half)
                self.print("d_probs", d_probs[:, half:])
                self.sample_new(D[t], d_dist)
                # D[:] = float(input("D:")) + half
                delta = D[t].clone() - half
                self.print("D[t], delta", D[t], delta)
            p = p + delta
            self.print("new p", p)
            p = torch.clamp(p, min=0, max=M.size(1) - 1)

            # try:
            # A[:] = float(input("A:"))
            # except ValueError:
            # pass
            yield RecurrentState(
                a=A[t],
                v=self.critic(z1),
                h=h,
                p=p,
                d=D[t],
                dg=dg,
                a_probs=a_dist.probs,
                d_probs=d_dist.probs,
                dg_probs=d_gate.probs,
            )

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        state_sizes = astuple(self.state_sizes)
        return RecurrentState(*torch.split(hx, state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        args = [
            torch.round(100 * a)
            if type(a) is torch.Tensor and a.dtype == torch.float
            else a
            for a in args
        ]
        if self.debug:
            print(*args, **kwargs)

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()
