from collections import Hashable
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from agents import AgentOutputs, NNBase
from data_types import RecurrentState, RawAction, CompoundAction
from env import Obs
from layers import MultiEmbeddingBag, IntEncoding
from utils import astuple, init_


class Categorical(torch.distributions.Categorical):
    def log_prob(self, value: torch.Tensor):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        shape = value.shape
        value = value.view(-1, 1)
        # gather = log_pmf.gather(-1, value).squeeze(-1)
        R = torch.arange(value.size(0))
        log_pmf = log_pmf.view(-1, log_pmf.size(-1))
        log_prob = log_pmf[R, value.squeeze(-1)]  # deterministic
        return log_prob.view(shape)


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
class AgentConfig:
    action_embed_size: int = 75
    add_layer: bool = True
    conv_hidden_size: int = 100
    debug: bool = False
    gate_coef: float = 0.01
    globalized_critic: bool = False
    instruction_embed_size: int = 128
    kernel_size: int = 2
    num_edges: int = 1
    no_pointer: bool = False
    no_roll: bool = False
    no_scan: bool = False
    olsk: bool = False
    resources_hidden_size: int = 128
    stride: int = 1
    transformer: bool = False
    zeta_activation: bool = False


@dataclass
class Agent(NNBase):
    activation_name: str
    add_layer: bool
    entropy_coef: float
    action_space: spaces.MultiDiscrete
    conv_hidden_size: int
    debug: bool
    gate_coef: float
    globalized_critic: bool
    hidden_size: int
    kernel_size: int
    action_embed_size: int
    max_eval_lines: int
    normalize: bool
    no_pointer: bool
    no_roll: bool
    no_scan: bool
    num_edges: int
    observation_space: spaces.Dict
    olsk: bool
    resources_hidden_size: int
    stride: int
    instruction_embed_size: int
    transformer: bool
    zeta_activation: bool
    inf: float = 1e5

    def __hash__(self):
        return self.hash()

    def __post_init__(self):
        nn.Module.__init__(self)
        self.activation = eval(f"nn.{self.activation_name}()")
        self.obs_spaces = Obs(**self.observation_space.spaces)
        self.action_nvec = RawAction.parse(*self.action_space.nvec)
        self.obs_sections = get_obs_sections(self.obs_spaces)
        self.eval_lines = self.max_eval_lines
        self.train_lines = len(self.obs_spaces.lines.nvec)

        self.embed_instruction = MultiEmbeddingBag(
            self.obs_spaces.lines.nvec[0], embedding_dim=self.instruction_embed_size
        )
        self.gru = nn.GRU(self.get_gru_in_size(), self.hidden_size)
        self.initial_hxs = nn.Parameter(
            torch.randn(self.action_embed_size), requires_grad=True
        )

        self.gru.reset_parameters()

        zeta_input_size = self.z1_size + self.instruction_embed_size
        instruction_encoder_in_size = self.instruction_embed_size + self.z1_size

        self.encode_G = nn.GRU(
            instruction_encoder_in_size,
            self.instruction_embed_size,
            bidirectional=True,
            batch_first=True,
        )
        self.initial_instruction_encoder_hxs = nn.Parameter(
            torch.randn(self.instruction_embed_size), requires_grad=True
        )
        self.encode_G.reset_parameters()

        self.embed_action = MultiEmbeddingBag(
            self.obs_spaces.partial_action.nvec,
            embedding_dim=self.action_embed_size,
        )

        extrinsic_nvec = self.action_nvec.a
        self.actor_logits_shape = len(extrinsic_nvec), max(extrinsic_nvec)
        num_actor_logits = int(np.prod(self.actor_logits_shape))
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))

        compound_action_size = CompoundAction.input_space().nvec.size

        d, h, w = self.obs_spaces.obs.shape
        self.obs_dim = d
        self.kernel_size = min(d, self.kernel_size)
        self.padding = optimal_padding(h, self.kernel_size, self.stride) + 1
        self.embed_resources = nn.Sequential(
            IntEncoding(self.resources_hidden_size),
            nn.Flatten(),
            self.init_(
                nn.Linear(2 * self.resources_hidden_size, self.resources_hidden_size)
            ),
            self.activation,
        )
        self.z_size = self.hidden_size if self.add_layer else zeta_input_size

        self.zeta = nn.Sequential(
            self.init_(
                nn.Linear(
                    zeta_input_size,
                    self.hidden_size,
                )
            ),
            self.activation,
        )
        if self.zeta_activation:
            self.zeta = nn.Sequential(self.activation, self.zeta)
        self.upsilon = self.build_upsilon()
        self.beta = self.build_beta()
        self.d_gate = self.build_d_gate()

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
            self.init_(
                nn.Linear(conv_out ** 2 * self.conv_hidden_size, self.conv_hidden_size)
            ),
        )
        if self.normalize:
            self.conv = nn.Sequential(nn.BatchNorm2d(d), self.conv)

        self.actor = self.init_(nn.Linear(self.z_size, num_actor_logits))

        critic_in_size = self.z_size

        if self.globalized_critic:
            critic_in_size = self.z1_size + 2 * self.instruction_embed_size
            if self.add_layer:
                self.eta = nn.Sequential(
                    self.init_(
                        nn.Linear(
                            critic_in_size,
                            self.hidden_size,
                        )
                    ),
                    self.activation,
                )
                critic_in_size = self.hidden_size

        self.critic = self.init_(nn.Linear(critic_in_size, 1))

        self.state_sizes = RecurrentState(
            a=1,
            a_probs=num_actor_logits,
            d=1,
            d_probs=(self.d_space()),
            h=self.hidden_size,
            p=1,
            v=1,
            dg_probs=2,
            dg=1,
        )

    def get_gru_in_size(self):
        return self.instruction_embed_size + self.action_embed_size

    def build_d_gate(self):
        return self.init_(nn.Linear(self.z_size, 2))

    def build_beta(self):
        in_size = (
            2 if self.no_roll or self.no_scan else 1
        ) * self.instruction_embed_size
        out_size = self.num_edges * self.d_space() if self.no_scan else self.num_edges
        return nn.Sequential(self.init_(nn.Linear(in_size, out_size)))

    @property
    def z1_size(self):
        return (
            self.conv_hidden_size
            + self.resources_hidden_size
            + self.action_embed_size
            + self.hidden_size
        )

    def build_P(self, p, G, R):
        N = p.size(0)
        G = G.view(N, self.nl, 2, -1)
        B = self.beta(G).sigmoid()
        # B = B * mask[p, R]
        f, b = torch.unbind(B, dim=-2)
        B = torch.stack([f, b.flip(-2)], dim=-2)
        B = B.view(N, 2 * self.nl, self.num_edges)

        last = torch.zeros(2 * self.nl, device=p.device)
        last[-1] = 1
        last = last.view(1, -1, 1)

        B = (1 - last).flip(-2) * B  # this ensures the first B is 0
        zero_last = (1 - last) * B
        B = zero_last + last  # this ensures that the last B is 1
        C = torch.cumprod(1 - torch.roll(zero_last, shifts=1, dims=-2), dim=-2)
        P = B * C
        P = P.view(N, self.nl, 2, self.num_edges)
        f, b = torch.unbind(P, dim=-2)

        return torch.cat([b.flip(-2), f], dim=-2)

    def build_upsilon(self):
        return self.init_(nn.Linear(self.z_size, self.num_edges))

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

    def forward(
        self, inputs, rnn_hxs, masks, deterministic=False, action=None, **kwargs
    ):
        N, dim = inputs.shape

        dists = RawAction.parse(None, None, None, None)
        if action is None:
            action = RawAction.parse(None, None, None, None)
        else:
            action = RawAction.parse(*action.unbind(-1))
            action = replace(action, a=torch.stack(action.a, dim=-1))

        # parse non-action inputs
        state = Obs(*torch.split(inputs, self.obs_sections, dim=-1))
        state = replace(state, obs=state.obs.view(N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(N, *self.obs_spaces.lines.shape).long()
        line_mask = state.line_mask.view(N, self.nl)
        line_mask = F.pad(line_mask, [self.nl, 0], value=1)  # pad for backward mask
        line_mask = torch.stack(
            [torch.roll(line_mask, shifts=-i, dims=-1) for i in range(self.nl)], dim=0
        )
        # mask[:, :, 0] = 0  # prevent self-loops
        # line_mask = line_mask.view(self.nl, N, 2, self.nl).transpose(2, 3).unsqueeze(-1)

        # build memory
        M = self.embed_instruction(
            lines.view(-1, self.obs_spaces.lines.nvec[0].size)
        ).view(N, -1, self.instruction_embed_size)
        p = state.ptr.long().flatten()
        R = torch.arange(N, device=p.device)
        rolled = torch.stack(
            [torch.roll(M, shifts=-i, dims=1) for i in range(self.nl)], dim=0
        )[p, R]

        x = self.conv(state.obs)
        resources = self.embed_resources(state.resources)
        embedded_action = self.embed_action(
            state.partial_action.long()
        )  # +1 to deal with negatives
        m = self.build_m(M, R, p)
        h, rnn_hxs = self._forward_gru(
            torch.cat([m, embedded_action], dim=-1), rnn_hxs, masks
        )
        z1 = torch.cat([x, resources, embedded_action, h], dim=-1)

        _z = z1.unsqueeze(1).expand(-1, rolled.size(1), -1)
        rolled = torch.cat([rolled, _z], dim=-1)
        G, _ = self.encode_G(rolled)

        ones = self.ones.expand_as(R)
        P = self.build_P(p, G, R)
        z = torch.cat([z1, m], dim=-1)
        if self.add_layer:
            z = self.zeta(z)
        zc = z
        if self.globalized_critic:
            zc = torch.cat([z1, G[R, p]], dim=-1)
            if self.add_layer:
                zc = self.eta(zc)

        self.print("p", p)

        a_logits = self.actor(z).view(-1, *self.actor_logits_shape)
        mask = state.action_mask.view(-1, *self.actor_logits_shape)
        mask = mask * -self.inf
        dists = replace(dists, a=Categorical(logits=a_logits + mask))

        self.print("a_probs", dists.a.probs)

        if action.a is None:
            a = dists.a.sample()
            action = replace(action, a=a)

        # while True:
        #     try:
        #         action = replace(
        #             action,
        #             a=float(input("a:")) * torch.ones_like(action.a),
        #         )
        #         break
        #     except ValueError:
        #         pass

        more_than_1_line = (1 - line_mask[p, R]).sum(-1) > 1
        dg, dg_dist = self.get_dg(
            can_open_gate=more_than_1_line,
            ones=ones,
            z=z,
        )
        dists = replace(dists, dg=dg_dist)
        if action.dg is None:
            action = replace(action, dg=dg)
            # if can_open_gate.item():
            #     while True:
            #         try:
            #             action = replace(
            #                 action,
            #                 dg=float(input("dg:")) * torch.ones_like(action.dg),
            #             )
            #             break
            #         except ValueError:
            #             pass

        delta, delta_dist = self.get_delta(
            P=P,
            dg=action.dg,
            line_mask=line_mask[p, R],
            ones=ones,
            z=z,
        )
        dists = replace(dists, delta=delta_dist)

        if action.delta is None:
            action = replace(action, delta=delta)
            # if action.dg.item():
            #     while True:
            #         try:
            #             print(self.nl)
            #             action = replace(
            #                 action,
            #                 delta=(float(input("delta:")) + self.nl)
            #                 * torch.ones_like(action.delta),
            #             )
            #             break
            #         except ValueError:
            #             pass

        delta = action.delta.clone() - self.nl
        self.print("action.delta, delta", action.delta, delta)

        if action.ptr is None:
            action = replace(action, ptr=p + delta)

        def compute_metric(raw: RawAction):
            raw = astuple(replace(raw, a=raw.a.sum(1)))
            return sum([x for x in raw if x is not None])

        action_log_probs = RawAction(
            *[
                None if dist is None else dist.log_prob(x)
                for dist, x in zip(astuple(dists), astuple(action))
            ],
        )
        entropy = RawAction(
            *[None if dist is None else dist.entropy() for dist in astuple(dists)]
        )
        aux_loss = -self.entropy_coef * compute_metric(entropy).mean()
        value = self.critic(zc)
        action = torch.cat(
            astuple(
                replace(
                    action,
                    dg=action.dg.unsqueeze(-1),
                    delta=action.delta.unsqueeze(-1),
                    ptr=action.ptr.unsqueeze(-1),
                )
            ),
            dim=-1,
        )

        # self.action_space.contains(action.numpy().squeeze(0))
        return AgentOutputs(
            value=value,
            action=action,
            action_log_probs=compute_metric(action_log_probs),
            aux_loss=aux_loss,
            dist=None,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    @staticmethod
    def build_m(M, R, p):
        return M[R, p]

    def get_delta(self, P, dg, line_mask, ones, z):
        u = self.upsilon(z).softmax(dim=-1)
        self.print("u", u)
        d_probs = (P @ u.unsqueeze(-1)).squeeze(-1)
        self.print("d_probs", d_probs.view(d_probs.size(0), 2, -1))
        unmask = 1 - line_mask
        masked = d_probs * unmask
        masked = masked / (masked + 1 - dg.unsqueeze(-1)).sum(-1, keepdim=True)
        self.print("masked", masked.view(masked.size(0), 2, -1))
        delta_dist = gate(dg.unsqueeze(-1), masked, ones * self.nl)
        # self.print("masked", Categorical(probs=masked).probs)
        self.print(
            "dists.delta", delta_dist.probs.view(delta_dist.probs.size(0), 2, -1)
        )
        delta = delta_dist.sample()
        return delta, delta_dist

    def get_dg(self, can_open_gate, ones, z):
        d_logits = self.d_gate(z)
        dg_probs = F.softmax(d_logits, dim=-1)
        can_open_gate = can_open_gate.long().unsqueeze(-1)
        dg_dist = gate(can_open_gate, dg_probs, ones * 0)
        self.print("dg prob", dg_dist.probs[:, 1])
        dg = dg_dist.sample()
        return dg, dg_dist

    def get_value(self, inputs, rnn_hxs, masks):
        return self.forward(inputs, rnn_hxs, masks).value

    def hash(self):
        return hash(tuple(x for x in astuple(self) if isinstance(x, Hashable)))

    def init_(self, m):
        return init_(m, nn.ReLU)

    @property
    def is_recurrent(self):
        return True

    @property
    def nl(self):
        return len(self.obs_spaces.lines.nvec)

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
        return self.hidden_size
