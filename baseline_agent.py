from collections import Hashable
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from agents import AgentOutputs
from data_types import RecurrentState, RawAction
from env import Obs
from layers import MultiEmbeddingBag, IntEncoding
from transformer import TransformerModel
from utils import astuple, init


class Categorical(torch.distributions.Categorical):
    def log_prob(self, value: torch.Tensor):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        # gather = log_pmf.gather(-1, value).squeeze(-1)
        R = torch.arange(value.size(0))
        return log_pmf[R, value.squeeze(-1)]  # deterministic


def optimal_padding(h, kernel, stride):
    n = np.ceil((h - kernel) / stride + 1)
    return int(np.ceil((stride * (n - 1) + kernel - h) / 2))


def conv_output_dimension(h, padding, kernel, stride, dilation=1):
    return int(1 + (h + 2 * padding - dilation * (kernel - 1) - 1) / stride)


def get_obs_sections(obs_spaces):
    return [int(np.prod(s.shape)) for s in astuple(obs_spaces)]


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return Categorical(probs=g * new + (1 - g) * old)


@dataclass
class Agent(nn.Module):
    entropy_coef: float
    action_space: spaces.MultiDiscrete
    conv_hidden_size: int
    debug: bool
    hidden_size: int
    kernel_size: int
    lower_embed_size: int
    max_eval_lines: int
    next_actions_embed_size: int
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
            self.obs_spaces.partial_action.nvec,
            embedding_dim=self.lower_embed_size,
        )
        self.embed_next_action = nn.Embedding(
            self.obs_spaces.next_actions.nvec[0],
            embedding_dim=self.next_actions_embed_size,
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
        self.actor = init_(nn.Linear(self.hidden_size, action_nvec.a))
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))

        d, h, w = self.obs_spaces.obs.shape
        self.obs_dim = d
        self.nl = len(self.obs_spaces.lines.nvec)
        self.kernel_size = min(d, self.kernel_size)
        self.padding = optimal_padding(h, self.kernel_size, self.stride) + 1
        self.embed_resources = nn.Sequential(
            IntEncoding(self.resources_hidden_size),
            nn.Flatten(),
            init_(
                nn.Linear(2 * self.resources_hidden_size, self.resources_hidden_size)
            ),
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
            + self.next_actions_embed_size * len(self.obs_spaces.next_actions.nvec)
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

        conv_out = conv_output_dimension(h, self.padding, self.kernel_size, self.stride)
        self.conv = nn.Sequential(
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
                nn.Linear(conv_out ** 2 * self.conv_hidden_size, self.conv_hidden_size)
            ),
        )
        self.conv_bias = nn.Parameter(torch.zeros(self.conv_hidden_size))
        self.critic = init_(nn.Linear(self.hidden_size, 1))

        last = torch.zeros(2 * self.nl)
        last[-1] = 1
        last = last.view(1, -1, 1)
        self.register_buffer("last", last)

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

    def build_P(self, p, M, R):
        N = p.size(0)

        rolled = torch.stack(
            [torch.roll(M, shifts=-i, dims=1) for i in range(self.nl)], dim=0
        )

        G, _ = self.task_encoder(rolled[p, R])
        G = G.view(N, self.nl, 2, -1)
        B = self.beta(G).sigmoid()
        # B = B * mask[p, R]
        f, b = torch.unbind(B, dim=-2)
        B = torch.stack([f, b.flip(-2)], dim=-2)
        B = B.view(N, 2 * self.nl, self.num_edges)
        B = (1 - self.last).flip(-2) * B  # this ensures the first B is 0
        zero_last = (1 - self.last) * B
        B = zero_last + self.last  # this ensures that the last B is 1
        C = torch.cumprod(1 - torch.roll(zero_last, shifts=1, dims=-2), dim=-2)
        P = B * C
        P = P.view(N, self.nl, 2, self.num_edges)
        f, b = torch.unbind(P, dim=-2)
        return torch.cat([b.flip(-2), f], dim=-2)

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
        self.state_sizes = replace(self.state_sizes, d_probs=self.d_space())
        self.obs_spaces = Obs(**self.obs_spaces)
        yield self
        self.obs_spaces = obs_spaces
        self.obs_sections = obs_sections
        self.state_sizes = state_sizes
        self.train_lines = train_lines

    # noinspection PyMethodOverriding
    def forward(
        self, inputs, rnn_hxs, masks, deterministic=False, action=None, **kwargs
    ):
        N, dim = inputs.shape

        dists = RawAction(None, None, None, None)
        if action is None:
            action = RawAction(None, None, None, None)
        else:
            action = RawAction(*action.unbind(-1))

        # parse non-action inputs
        state = Obs(*torch.split(inputs, self.obs_sections, dim=-1))
        state = replace(state, obs=state.obs.view(N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(N, *self.obs_spaces.lines.shape).long()

        # build memory
        M = self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            N, -1, self.task_embed_size
        )
        p = state.ptr.long().flatten()
        R = torch.arange(N, device=p.device)
        ones = self.ones.expand_as(R)
        P = self.build_P(p, M, R)
        m = M[R, p]
        self.print("p", p)

        h1 = self.conv(state.obs)
        resources = self.embed_resources(state.resources)
        next_actions = self.embed_next_action(state.next_actions.long()).view(N, -1)
        embedded_lower = self.embed_lower(
            state.partial_action.long()
        )  # +1 to deal with negatives
        zeta1_input = torch.cat(
            [m, h1, resources, embedded_lower, next_actions], dim=-1
        )
        z1 = F.relu(self.zeta1(zeta1_input))

        value = self.critic(z1)

        a_logits = self.actor(z1) - state.action_mask * 1e10
        dists = replace(dists, a=Categorical(logits=a_logits))

        self.print("a_probs", dists.a.probs)

        if action.a is None:
            a = dists.a.sample()
            action = replace(action, a=a)

        d_logits = self.d_gate(zeta1_input)
        d_probs = F.softmax(d_logits, dim=-1)
        can_open_gate = state.can_open_gate[R, action.a].long().unsqueeze(-1)
        dists = replace(dists, dg=gate(can_open_gate, d_probs, ones * 0))

        if action.dg is None:
            action = replace(action, dg=dists.dg.sample())

        u = self.upsilon(zeta1_input).softmax(dim=-1)
        self.print("u", u)
        d_probs = (P @ u.unsqueeze(-1)).squeeze(-1)

        self.print("dg prob", dists.dg.probs[:, 1])

        dists = replace(
            dists, delta=gate(action.dg.unsqueeze(-1), d_probs, ones * self.nl)
        )
        self.print("d_probs", d_probs[:, self.nl :])

        if action.delta is None:
            action = replace(action, delta=dists.delta.sample())

        delta = action.delta.clone() - self.nl
        self.print("action.delta, delta", action.delta, delta)

        if action.ptr is None:
            action = replace(action, ptr=p + delta)

        action = replace(
            action,
            dg=torch.zeros_like(action.dg),
            delta=torch.zeros_like(action.delta),
            ptr=torch.zeros_like(action.ptr),
        )
        action_log_probs = dists.a.log_prob(action.a).unsqueeze(-1)  # TODO
        entropy = sum(
            [dist.entropy() for dist in astuple(dists) if dist is not None]
        ).mean()
        return AgentOutputs(
            value=value,
            action=torch.stack(astuple(action), dim=-1),
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=None,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    def get_value(self, inputs, rnn_hxs, masks):
        return self.forward(inputs, rnn_hxs, masks).value

    @property
    def is_recurrent(self):
        return False

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

    @property
    def recurrent_hidden_state_size(self):
        return 1
