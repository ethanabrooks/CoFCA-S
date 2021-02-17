from collections import Hashable
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from agents import AgentOutputs, NNBase
from data_types import RawAction, RecurrentState, Obs
from layers import MultiEmbeddingBag
from utils import astuple, init_


class Categorical(torch.distributions.Categorical):
    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

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


def apply_gate(gate, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return Categorical(probs=gate * new + (1 - gate) * old)


@dataclass
class AgentConfig:
    action_embed_size: int = 75
    add_actor_layer: bool = False
    add_beta_layer: bool = False
    add_critic_layer: bool = False
    add_gate_layer: bool = False
    add_upsilon_layer: bool = False
    b_dot_product: bool = False
    bidirectional_beta_inputs: bool = False
    conv_hidden_size: int = 100
    debug: bool = False
    destroyed_unit_embed_size: int = 100
    gate_coef: float = 0.01
    instruction_embed_size: int = 128
    kernel_size: int = 2
    num_edges: int = 3
    no_pointer: bool = False
    no_roll: bool = False
    no_scan: bool = False
    olsk: bool = False
    resources_hidden_size: int = 128
    stride: int = 1
    transformer: bool = False


# noinspection PyPep8Naming
@dataclass
class Agent(NNBase):
    activation_name: str
    add_actor_layer: bool
    add_beta_layer: bool
    add_critic_layer: bool
    add_gate_layer: bool
    add_upsilon_layer: bool
    b_dot_product: bool
    bidirectional_beta_inputs: bool
    entropy_coef: float
    action_space: spaces.MultiDiscrete
    conv_hidden_size: int
    debug: bool
    destroyed_unit_embed_size: int
    gate_coef: float
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
        self.train_lines = len(self.obs_spaces.instructions.nvec)

        self.resources_hidden_size = (
            self.resources_hidden_size // 2 * 2
        )  # make divisible by 2

        self.embed_destroyed_unit = nn.Embedding(
            self.obs_spaces.destroyed_unit.n,
            embedding_dim=self.destroyed_unit_embed_size,
        )
        self.embed_instruction = nn.Embedding(
            self.obs_spaces.instructions.nvec[0],
            embedding_dim=self.instruction_embed_size,
        )

        self.gru = nn.GRU(
            self.gru_in_size,
            self.hidden_size,
        )
        self.gru.reset_parameters()

        self.g_gru = self.build_g_gru()

        self.encode_G = self.build_encode_G()
        self.initial_instruction_encoder_hxs = nn.Parameter(
            torch.randn(self.rolled_size), requires_grad=True
        )
        self.encode_G.reset_parameters()

        self.embed_action = MultiEmbeddingBag(
            self.obs_spaces.partial_action.nvec,
            embedding_dim=self.action_embed_size,
        )

        extrinsic_nvec = self.action_nvec.extrinsic
        self.actor_logits_shape = len(extrinsic_nvec), max(extrinsic_nvec)
        num_actor_logits = int(np.prod(self.actor_logits_shape))
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))

        self.gate_openers_shape = self.obs_spaces.gate_openers.nvec.shape

        d, h, w = self.obs_spaces.obs.shape
        self.obs_dim = d
        self.kernel_size = min(d, self.kernel_size)
        self.padding = optimal_padding(h, self.kernel_size, self.stride) + 1
        # self.embed_resources = nn.Sequential(
        #     IntEncoding(self.resources_hidden_size),
        #     nn.Flatten(),
        #     self.init_(
        #         nn.Linear(2 * self.resources_hidden_size, self.resources_hidden_size)
        #     ),
        #     self.activation,
        # )
        self.z_size = self.hidden_size

        self.upsilon = self.build_upsilon()
        self.beta = self.build_beta()
        self.gate = self.build_gate()

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

        if self.add_actor_layer:
            self.actor = nn.Sequential(
                self.init_(nn.Linear(self.za_size, self.hidden_size)),
                self.activation,
                self.init_(nn.Linear(self.hidden_size, num_actor_logits)),
            )
        else:
            self.actor = self.init_(nn.Linear(self.za_size, num_actor_logits))

        if self.add_critic_layer:
            self.critic = nn.Sequential(
                self.init_(nn.Linear(self.zg_size, self.hidden_size)),
                self.activation,
                self.init_(nn.Linear(self.hidden_size, 1)),
            )
        else:
            self.critic = self.init_(nn.Linear(self.zg_size, 1))

        self.state_sizes = RecurrentState(
            a=1,
            a_probs=num_actor_logits,
            d=1,
            d_probs=self.delta_size,
            h=self.hidden_size,
            p=1,
            v=1,
            dg_probs=2,
            dg=1,
        )

    @property
    def gru_in_size(self):
        return (
            self.conv_hidden_size
            + self.action_embed_size
            + self.destroyed_unit_embed_size
        )

    @property
    def rolled_size(self):
        return self.instruction_embed_size

    @property
    def za_size(self):
        return self.z_size + self.instruction_embed_size

    @property
    def zg_size(self):
        return self.za_size + 4 * self.rolled_size * (
            self.num_edges if self.b_dot_product else 1
        )

    def build_g_gru(self):
        gru = nn.GRU(2 * self.rolled_size, self.hidden_size)
        gru.reset_parameters()
        return gru

    def build_encode_G(self):
        return nn.GRU(
            self.rolled_size,
            self.rolled_size * (self.num_edges if self.b_dot_product else 1),
            bidirectional=True,
            batch_first=True,
        )

    def build_beta(self):
        beta_in_size = self.zg_size + (
            0
            if self.b_dot_product
            else self.rolled_size * (2 if self.bidirectional_beta_inputs else 1)
        )
        beta_out_size = self.num_edges * (
            (2 if self.bidirectional_beta_inputs else 1) * self.rolled_size
            if self.b_dot_product
            else 1
        )
        if self.add_beta_layer:
            return nn.Sequential(
                self.init_(nn.Linear(beta_in_size, self.hidden_size)),
                self.activation,
                self.init_(nn.Linear(self.hidden_size, beta_out_size)),
            )
        else:
            return self.init_(nn.Linear(beta_in_size, beta_out_size))

    def build_gate(self):
        if self.add_gate_layer:
            return nn.Sequential(
                self.init_(nn.Linear(self.zg_size, self.hidden_size)),
                self.activation,
                self.init_(nn.Linear(self.hidden_size, 2)),
            )
        else:
            return self.init_(nn.Linear(self.zg_size, 2))

    @staticmethod
    def build_m(M, R, p):
        return M[R, p]

    def build_upsilon(self):
        if self.add_upsilon_layer:
            return nn.Sequential(
                self.init_(nn.Linear(self.zg_size, self.hidden_size)),
                self.activation,
                self.init_(nn.Linear(self.hidden_size, 2 * self.num_edges)),
            )
        else:
            return self.init_(nn.Linear(self.zg_size, 2 * self.num_edges))

    @property
    def max_backward_jump(self):
        return self.train_lines

    @property
    def max_forward_jump(self):
        return self.train_lines - 1

    @property
    def delta_size(self):
        return self.max_backward_jump + 1 + self.max_forward_jump

    # noinspection PyAttributeOutsideInit
    @contextmanager
    def evaluating(self, eval_obs_space):
        obs_spaces = self.obs_spaces
        obs_sections = self.obs_sections
        state_sizes = self.state_sizes
        train_lines = self.train_lines
        self.obs_spaces = Obs(**eval_obs_space.spaces)
        self.obs_sections = get_obs_sections(self.obs_spaces)
        self.train_lines = len(self.obs_spaces.instructions.nvec)
        # noinspection PyProtectedMember
        self.state_sizes = replace(self.state_sizes, d_probs=self.delta_size)
        yield self
        self.obs_spaces = obs_spaces
        self.obs_sections = obs_sections
        self.state_sizes = state_sizes
        self.train_lines = train_lines

    def forward(self, inputs, rnn_hxs, masks, action=None):
        N, dim = inputs.shape

        dists = RawAction.parse(None, None, None, None, None)
        if action is None:
            action = RawAction.parse(None, None, None, None, None)
        else:
            action = RawAction.parse(*action.unbind(-1))
            action = replace(action, extrinsic=torch.stack(action.extrinsic, dim=-1))

        # action_rnn_hxs, g_rnn_hxs = self.split_rnn_hxs(rnn_hxs)

        # parse non-action inputs
        state = Obs(*torch.split(inputs, self.obs_sections, dim=-1))
        state = replace(state, obs=state.obs.view(N, *self.obs_spaces.obs.shape))
        instructions = state.instructions.view(
            N, *self.obs_spaces.instructions.shape
        ).long()
        instruction_mask = state.instruction_mask
        instruction_mask = self.get_instruction_mask(N, instruction_mask)
        # mask[:, :, 0] = 0  # prevent self-loops
        # line_mask = line_mask.view(self.nl, N, 2, self.nl).transpose(2, 3).unsqueeze(-1)

        # build memory
        M = self.embed_instruction(
            instructions.view(-1, self.obs_spaces.instructions.nvec[0].size)
        ).view(N, -1, self.instruction_embed_size)
        assert M.size(-1) == self.instruction_embed_size
        p1 = state.pointer1.long().flatten()
        p2 = state.pointer2.long().flatten()
        R = torch.arange(N, device=p1.device)
        rolled1 = self.get_rolled(M, R, p1)
        rolled2 = self.get_rolled(M, R, p2)
        for rolled in (rolled1, rolled2):
            if rolled is not None:
                assert rolled.size(-1) == self.rolled_size

        x = self.conv(state.obs)
        destroyed_unit = self.embed_destroyed_unit(state.destroyed_unit.long()).view(
            N, self.destroyed_unit_embed_size
        )
        embedded_action = self.embed_action(
            state.partial_action.long()
        )  # +1 to deal with negatives
        G, g1 = self.get_G_g(rolled1)
        G, g2 = self.get_G_g(rolled2)
        m1 = self.build_m(M, R, p1)
        m2 = self.build_m(M, R, p2)

        # ha, action_rnn_hxs, hg, g_rnn_hxs = self.update_hxs(
        #     embedded_action=embedded_action,
        #     destroyed_unit=destroyed_unit,
        #     action_rnn_hxs=action_rnn_hxs,
        #     g=g,
        #     g_rnn_hxs=g_rnn_hxs,
        #     masks=masks,
        # )

        # z = torch.cat([x, ha], dim=-1)
        z, rnn_hxs = self.forward_gru(
            destroyed_unit=destroyed_unit,
            embedded_action=embedded_action,
            m=m1,  # TODO
            masks=masks,
            rnn_hxs=rnn_hxs,
            x=x,
        )
        assert z.size(-1) == self.z_size

        za = torch.cat([z, m1], dim=-1)
        assert za.size(-1) == self.za_size
        # zg = self.get_zg(za, hg, za)
        zg = self.get_zg(za, g1, g2, za)
        assert zg.size(-1) == self.zg_size

        ones = self.ones.expand_as(R)
        P1 = self.get_P(p1, G, R, zg)
        P2 = self.get_P(p2, G, R, zg)

        self.print("p", p1)

        a_logits = self.actor(za).view(-1, *self.actor_logits_shape)
        mask = state.action_mask.view(-1, *self.actor_logits_shape)
        mask = mask * -self.inf
        dists = replace(dists, extrinsic=Categorical(logits=a_logits + mask))

        self.print("a_probs", dists.extrinsic.probs)

        if action.extrinsic is None:
            extrinsic = dists.extrinsic.sample()
            action = replace(action, extrinsic=extrinsic)

        # while True:
        #     try:
        #         action = replace(
        #             action,
        #             a=float(input("a:")) * torch.ones_like(action.a),
        #         )
        #         break
        #     except ValueError:
        #         pass

        gate_openers = state.gate_openers.view(-1, *self.gate_openers_shape)[R]
        matches = gate_openers == action.extrinsic.unsqueeze(1)
        assert isinstance(matches, torch.Tensor)
        more_than_1_line = (1 - instruction_mask[p1, R]).sum(-1) > 1
        # noinspection PyArgumentList
        can_open_gate = matches.all(-1).any(-1) * more_than_1_line
        gate, gate_dist = self.get_gate(
            can_open_gate=can_open_gate,
            ones=ones,
            z=zg,
        )
        dists = replace(dists, gate=gate_dist)
        if action.gate is None:
            action = replace(action, gate=gate)
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

        d_probs = self.get_delta_probs(G, P1, P2, zg)
        d, d_dist = self.get_delta(
            delta_probs=d_probs,
            dg=action.gate,
            line_mask=instruction_mask[p1, R],
            ones=ones,
        )
        dists = replace(dists, delta=d_dist)

        if action.delta is None:
            action = replace(action, delta=d)
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

        d = action.delta.clone() - self.max_backward_jump
        self.print("action.delta, delta", action.delta, d)

        if action.pointer1 is None:
            action = replace(action, pointer1=p1 + d)

        if action.pointer2 is None:
            action = replace(action, pointer2=torch.max(p2, action.pointer1))

        def compute_metric(raw: RawAction):
            raw = astuple(replace(raw, extrinsic=raw.extrinsic.sum(1)))
            return sum([y for y in raw if y is not None])

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
        value = self.critic(zg)
        action = torch.cat(
            astuple(
                replace(
                    action,
                    gate=action.gate.unsqueeze(-1),
                    delta=action.delta.unsqueeze(-1),
                    pointer1=action.pointer1.unsqueeze(-1),
                    pointer2=action.pointer2.unsqueeze(-1),
                )
            ),
            dim=-1,
        )

        # rnn_hxs = self.combine_rnn_hxs(action_rnn_hxs, g_rnn_hxs)
        # self.action_space.contains(action.numpy().squeeze(0))
        return AgentOutputs(
            value=value,
            action=action,
            action_log_probs=compute_metric(action_log_probs),
            aux_loss=aux_loss,
            dist=dists,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    def forward_gru(self, destroyed_unit, embedded_action, m, masks, rnn_hxs, x):
        y = torch.cat([x, destroyed_unit, embedded_action], dim=-1)
        z, rnn_hxs = self._forward_gru(y, rnn_hxs, masks, self.gru)
        return z, rnn_hxs

    def get_zg(self, z, g1, g2, za):
        return torch.cat(
            [z, g1.reshape(z.size(0), -1), g2.reshape(z.size(0), -1)], dim=-1
        )

    def split_rnn_hxs(self, rnn_hxs):
        return torch.split(rnn_hxs, self.hidden_size, dim=-1)

    @staticmethod
    def combine_rnn_hxs(action_rnn_hxs, g_rnn_hxs):
        return torch.cat([action_rnn_hxs, g_rnn_hxs], dim=-1)

    def get_instruction_mask(self, N, instruction_mask):
        line_mask = instruction_mask.view(N, self.instruction_length)
        line_mask = F.pad(
            line_mask, [self.max_backward_jump, 0], value=1
        )  # pad for backward mask
        line_mask = torch.stack(
            [
                torch.roll(line_mask, shifts=-i, dims=-1)
                for i in range(self.instruction_length)
            ],
            dim=0,
        )
        return line_mask

    def get_delta(self, delta_probs, dg, line_mask, ones):
        unmask = 1 - line_mask
        masked = unmask * delta_probs
        sum_zero = masked.sum(-1, keepdim=True) < 1 / self.inf
        masked = ~sum_zero * masked + sum_zero * unmask  # uniform distribution
        delta_dist = apply_gate(dg.unsqueeze(-1), masked, ones * self.max_backward_jump)
        # self.print("masked", Categorical(probs=masked).probs)
        self.print("dists.delta", delta_dist.probs.view(delta_dist.probs.size(0), -1))
        delta = delta_dist.sample()
        return delta, delta_dist

    def get_delta_probs(self, G, P1, P2, z):
        u = self.upsilon(z).softmax(dim=-1)
        self.print("u", u)
        P = torch.cat([P1, P2], dim=-1)
        delta_probs = (P @ u.unsqueeze(-1)).squeeze(-1)
        self.print("d_probs", delta_probs.view(delta_probs.size(0), -1))
        return delta_probs

    def get_gate(self, can_open_gate, ones, z):
        d_logits = self.gate(z)
        dg_probs = F.softmax(d_logits, dim=-1)
        can_open_gate = can_open_gate.long().unsqueeze(-1)
        dg_dist = apply_gate(can_open_gate, dg_probs, ones * 0)
        self.print("dg prob", dg_dist.probs[:, 1])
        dg = dg_dist.sample()
        return dg, dg_dist

    def get_G_g(self, rolled):
        G, g = self.encode_G(rolled)
        return G, g.transpose(0, 1)

    def get_rolled(self, M, R, p):
        return torch.stack(
            [torch.roll(M, shifts=-i, dims=1) for i in range(self.instruction_length)],
            dim=0,
        )[p, R]

    def get_P(self, p, G, R, zg):
        N = p.size(0)
        if self.bidirectional_beta_inputs:
            G = G.view(N, self.instruction_length, 2, -1)
            G = torch.cat([G, G.flip(-2)], dim=-1)
        if self.b_dot_product:
            G = G.view(N, self.instruction_length, 2, self.num_edges, -1)
            beta_out = self.beta(zg).view(N, 1, 1, self.num_edges, -1)
            b = torch.sum(beta_out * G, dim=-1)
        else:
            G = G.view(N, self.instruction_length, 2, -1)
            expanded = zg.view(N, 1, 1, self.zg_size).expand(
                -1, G.size(1), G.size(2), -1
            )
            b = self.beta(torch.cat([G, expanded], dim=-1))

        B = b.sigmoid()  # N, nl, 2, ne
        # B = B * mask[p, R]
        f, b = torch.unbind(B, dim=-2)
        B = torch.stack([f, b.flip(-2)], dim=-2)
        B = B.view(N, 2 * self.instruction_length, self.num_edges)

        last = torch.zeros(2 * self.instruction_length, device=p.device)
        last[-1] = 1
        last = last.view(1, -1, 1)

        B = (1 - last).flip(-2) * B  # this ensures the first B is 0
        zero_last = (1 - last) * B
        B = zero_last + last  # this ensures that the last B is 1
        C = torch.cumprod(1 - torch.roll(zero_last, shifts=1, dims=-2), dim=-2)
        P = B * C
        P = P.view(N, self.instruction_length, 2, self.num_edges)
        f, b = torch.unbind(P, dim=-2)

        return torch.cat([b.flip(-2), f], dim=-2)

    def get_value(self, inputs, rnn_hxs, masks):
        return self.forward(inputs, rnn_hxs, masks).value

    def hash(self):
        return hash(tuple(x for x in astuple(self) if isinstance(x, Hashable)))

    def init_(self, m):
        return init_(m, type(self.activation))

    @property
    def is_recurrent(self):
        return True

    @property
    def instruction_length(self):
        return len(self.obs_spaces.instructions.nvec)

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
