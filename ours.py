from collections import namedtuple
from contextlib import contextmanager
from contextlib import contextmanager
from dataclasses import fields
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch import Tensor

import data_types
from distributions import FixedCategorical
import env
from networks import MultiEmbeddingBag, IntEncoding
from transformer import TransformerModel
from utils import astuple, asdict, init, replace

# necessary for JIT unfortunately
ParsedInput = namedtuple(
    "ParsedInput", [f.name for f in fields(data_types.ParsedInput)]
)
RecurrentState = namedtuple(
    "RecurrentState", [f.name for f in fields(data_types.RecurrentState)]
)
RawAction = namedtuple("RawAction", [f.name for f in fields(data_types.RawAction)])
Obs = namedtuple("Obs", [f.name for f in fields(env.Obs)])


def optimal_padding(h, kernel, stride):
    n = np.ceil((h - kernel) / stride + 1)
    return int(np.ceil((stride * (n - 1) + kernel - h) / 2))


def conv_output_dimension(h, padding, kernel, stride, dilation=1):
    return int(1 + (h + 2 * padding - dilation * (kernel - 1) - 1) / stride)


def get_obs_sections(obs_spaces):
    return [int(np.prod(s.shape)) for s in astuple(obs_spaces)]


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return g * new + (1 - g) * old


# noinspection PyArgumentList
class Recurrence(torch.jit.ScriptModule):
    def __init__(
        self,
        action_space: spaces.MultiDiscrete,
        conv_hidden_size: int,
        debug: bool,
        hidden_size: int,
        kernel_size: int,
        lower_embed_size: int,
        max_eval_lines: int,
        next_actions_embed_size: int,
        no_pointer: bool,
        no_roll: bool,
        no_scan: bool,
        num_edges: int,
        observation_space: spaces.Dict,
        olsk: bool,
        resources_hidden_size: int,
        stride: int,
        task_embed_size: int,
        transformer: bool,
    ):
        self.action_space = action_space
        self.conv_hidden_size = conv_hidden_size
        self.debug = debug
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.lower_embed_size = lower_embed_size
        self.max_eval_lines = max_eval_lines
        self.next_actions_embed_size = next_actions_embed_size
        self.no_pointer = no_pointer
        self.no_roll = no_roll
        self.no_scan = no_scan
        self.num_edges = num_edges
        self.observation_space = observation_space
        self.olsk = olsk
        self.resources_hidden_size = resources_hidden_size
        self.stride = stride
        self.task_embed_size = task_embed_size
        self.transformer = transformer
        super().__init__()
        x1, x2, x3, x4, x5, x6, x7, x8 = list(self.observation_space.spaces.values())
        self.obs_spaces = Obs(x1, x2, x3, x4, x5, x6, x7, x8)
        self.action_size = self.action_space.nvec.size

        self.obs_sections = get_obs_sections(self.obs_spaces)
        self.obs_size = sum(self.obs_sections)
        self.input_sections = ParsedInput(obs=self.obs_size, actions=self.action_size)

        self.eval_lines = self.max_eval_lines
        self.train_lines = len(self.obs_spaces.lines.nvec)

        action_nvec = RawAction(*map(int, self.action_space.nvec))

        self.embed_task = MultiEmbeddingBag(
            [int(x) for x in self.obs_spaces.lines.nvec[0]],
            embedding_dim=self.task_embed_size,
        )
        self.embed_lower = MultiEmbeddingBag(
            [int(x) for x in self.obs_spaces.partial_action.nvec],
            embedding_dim=self.lower_embed_size,
        )
        self.embed_next_action = torch.jit.script(
            nn.Embedding(
                int(self.obs_spaces.next_actions.nvec[0]),
                embedding_dim=self.next_actions_embed_size,
            )
        )
        self.task_encoder = (
            torch.jit.script(
                TransformerModel(
                    ntoken=self.num_edges * self.d_space(),
                    ninp=self.task_embed_size,
                    nhid=self.task_embed_size,
                )
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
        self.actor = torch.jit.script(init_(nn.Linear(self.hidden_size, action_nvec.a)))
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))

        self.d, self.h, self.w = d, h, w = self.obs_spaces.obs.shape
        self.nl, self.line_dim = self.obs_spaces.lines.shape
        self.obs_dim = d
        self.kernel_size = min(d, self.kernel_size)
        self.padding = optimal_padding(h, self.kernel_size, self.stride) + 1
        self.embed_resources = torch.jit.script(
            nn.Sequential(
                IntEncoding(d_model=self.resources_hidden_size),
                nn.Flatten(),
                init_(
                    nn.Linear(
                        2 * self.resources_hidden_size, self.resources_hidden_size
                    )
                ),
                nn.ReLU(),
            )
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
            + self.next_actions_embed_size * len(self.obs_spaces.next_actions.nvec)
        )
        self.zeta1 = torch.jit.script(
            init_(nn.Linear(zeta1_input_size, self.hidden_size))
        )
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
        self.upsilon = torch.jit.script(self.upsilon)
        self.beta = torch.jit.script(self.beta)
        self.d_gate = torch.jit.script(init_(nn.Linear(zeta1_input_size, 2)))

        conv_out = conv_output_dimension(h, self.padding, self.kernel_size, self.stride)
        self.conv = torch.jit.script(
            nn.Sequential(
                nn.Conv2d(
                    d,
                    self.conv_hidden_size,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                ),
                nn.ReLU(),
                nn.Flatten(),
                init_(
                    nn.Linear(
                        conv_out ** 2 * self.conv_hidden_size, self.conv_hidden_size
                    )
                ),
            )
        )
        self.conv_bias = nn.Parameter(torch.zeros(self.conv_hidden_size))
        self.critic = torch.jit.script(init_(nn.Linear(self.hidden_size, 1)))
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

    # noinspection PyPep8Naming
    @torch.jit.script_method
    def inner_loop(self, raw_inputs, rnn_hxs):
        recurrent_states = torch.jit.annotate(List[RecurrentState], [])
        T, N, dim = raw_inputs.shape
        x1, x2 = torch.split(raw_inputs, self.input_sections, dim=-1)
        inputs = ParsedInput(x1, x2)

        # parse non-action inputs
        x1, x2, x3, x4, x5, x6, x7, x8 = torch.split(
            inputs.obs, self.obs_sections, dim=-1
        )
        state = Obs(x1, x2, x3, x4, x5, x6, x7, x8)
        # state = state._replace(obs=state.obs.view(T, N, *self.obs_spaces.obs.shape))
        lines = state.lines[0].view(N, self.nl, self.line_dim).long()
        # mask = state.mask[0].view(N, nl)
        # mask = F.pad(mask, [0, nl], value=1)  # pad for backward mask
        # mask = torch.stack(
        #     [torch.roll(mask, shifts=-i, dims=1) for i in range(nl)], dim=0
        # )
        # mask[:, :, 0] = 0  # prevent self-loops
        # mask = mask.view(nl, N, 2, nl).transpose(2, 3).unsqueeze(-1)

        # build memory
        M = self.embed_task(lines.view(-1, self.line_dim)).view(
            N, -1, self.task_embed_size
        )
        rolled = torch.stack(
            [torch.roll(M, shifts=-i, dims=1) for i in range(self.nl)], dim=0
        )
        # first = torch.zeros(2 * nl, device=raw_inputs.device)
        # first[0] = 0.1
        # first = first.view(1, -1, 1)
        last = torch.zeros(2 * self.nl, device=rnn_hxs.device)
        last[-1] = 1
        last = last.view(1, -1, 1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs.squeeze(0))

        p = hx.p.long().squeeze(-1)
        h = hx.h
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        x1, x2, x3, x4 = inputs.actions.unbind(dim=2)
        actions = RawAction(x1, x2, x3, x4)
        prev_a = hx.a.view(1, N)
        A = torch.cat([actions.a, prev_a], dim=0).long()
        D = actions.delta.long()
        DG = actions.dg.long()

        for t in range(T):
            # if self.no_pointer:
            #     _, G = self.task_encoder(M)
            #     P = G.transpose(0, 1).reshape(N, -1)
            # if self.no_scan:
            #     if self.no_roll:
            #         H, _ = self.task_encoder(M)
            #     else:
            #         rolled = torch.cat(
            #             [torch.roll(M, shifts=-i, dims=1) for i in range(self.nl)],
            #             dim=0,
            #         )
            #         _, H = self.task_encoder(rolled)
            #     H = H.transpose(0, 1).reshape(self.nl, N, -1)
            #     P = self.beta(H).view(self.nl, N, -1, self.num_edges).softmax(2)
            # elif self.transformer:
            #     P = self.task_encoder(M.transpose(0, 1))
            #     # _, P = (
            #     #     self.task_encoder(M.transpose(0, 1))
            #     #     .view(self.nl, N, -1, self.num_edges)
            #     #     .softmax(2)
            #     # )
            # else:
            #     if self.no_roll:
            #         G, _ = self.task_encoder(M)
            #         G = torch.cat(
            #             [
            #                 G.unsqueeze(1).expand(-1, self.nl, -1, -1),
            #                 G.unsqueeze(2).expand(-1, -1, self.nl, -1),
            #             ],
            #             dim=-1,
            #         ).transpose(0, 1)
            #     else:
            G, _ = self.task_encoder(rolled[p, R])
            G = G.view(N, self.nl, 2, -1)
            B = self.beta(G).sigmoid()
            # B = B * mask[p, R]
            f, b = torch.unbind(B, dim=-2)
            B = torch.stack([f, b.flip(-2)], dim=-2)
            B = B.view(N, 2 * self.nl, self.num_edges)
            B = (1 - last).flip(-2) * B  # this ensures the first B is 0
            zero_last = (1 - last) * B
            B = zero_last + last  # this ensures that the last B is 1
            C = torch.cumprod(1 - torch.roll(zero_last, shifts=1, dims=-2), dim=-2)
            P = B * C
            P = P.view(N, self.nl, 2, self.num_edges)
            f, b = torch.unbind(P, dim=-2)
            P = torch.cat([b.flip(-2), f], dim=-2)
            # noinspection PyArgumentList
            half = P.size(2) // 2 if self.no_scan else self.nl
            self.print("p", p)
            m = torch.cat([P, h], dim=-1) if self.no_pointer else M[R, p]

            h1 = self.conv(state.obs[t].view(N, self.d, self.h, self.w))
            resources = self.embed_resources(state.resources[t])
            next_actions = self.embed_next_action(state.next_actions[t].long()).view(
                N, -1
            )
            embedded_lower = self.embed_lower(
                state.partial_action[t].long()
            )  # +1 to deal with negatives
            zeta1_input = torch.cat(
                [m, h1, resources, embedded_lower, next_actions], dim=-1
            )
            z1 = F.relu(self.zeta1(zeta1_input))

            a_probs = F.softmax(self.actor(z1) - state.action_mask[t] * 1e10, dim=-1)
            self.print("action_mask", state.action_mask[t])
            self.sample_new(A[t], a_probs)

            self.print("a_probs", a_probs)

            d_probs = F.softmax(self.d_gate(zeta1_input), dim=-1)
            can_open_gate = state.can_open_gate[t, R, A[t]].long().unsqueeze(-1)
            dg_probs = gate(can_open_gate, d_probs, ones * 0)
            self.sample_new(DG[t], dg_probs)
            dg = DG[t].unsqueeze(-1).float()

            if self.olsk or self.no_pointer:
                h = self.upsilon(zeta1_input)
                u = self.beta(h).softmax(dim=-1)
                d_dist = gate(dg, u, ones)
                self.sample_new(D[t], d_dist)
                delta = D[t].clone() - 1
            else:
                u = self.upsilon(zeta1_input).softmax(dim=-1)
                self.print("u", u)
                d_probs = (P @ u.unsqueeze(-1)).squeeze(-1)

                self.print("dg prob", dg_probs[:, 1])
                self.print("dg", dg)
                d_probs = gate(dg, d_probs, ones * half)
                self.print("d_probs", d_probs[:, half:])
                self.sample_new(D[t], d_probs)
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
            recurrent_states += [
                RecurrentState(
                    a=A[t],
                    v=self.critic(z1),
                    h=h,
                    p=p,
                    d=D[t],
                    dg=dg,
                    a_probs=a_probs,
                    d_probs=d_probs,
                    dg_probs=d_probs,
                )
            ]
        return recurrent_states

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        state_sizes = torch.jit.annotate(
            Tuple[int, int, int, int, int, int, int, int, int],
            self.state_sizes,
        )
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = torch.split(hx, state_sizes, dim=-1)
        return RecurrentState(x1, x2, x3, x4, x5, x6, x7, x8, x9)

    @torch.jit.ignore
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
    @torch.jit.ignore
    def sample_new(x, probs):
        new = x < 0
        dist = FixedCategorical(probs=probs)
        x[new] = dist.sample()[new].flatten()
