import gc
import json
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

import ppo
import ppo.control_flow.multi_step.abstract_recurrence as abstract_recurrence
import ppo.control_flow.recurrence as recurrence
from ppo.agent import Agent
from ppo.control_flow.env import Action
from ppo.control_flow.lstm import LSTMCell
from ppo.control_flow.multi_step.env import Obs
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState",
    "a l d u ag dg p v h lh hy cy a_probs d_probs ag_probs dg_probs gru_gate P",
)

ParsedInput = namedtuple("ParsedInput", "obs actions")


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


class Recurrence(abstract_recurrence.Recurrence, recurrence.Recurrence):
    def __init__(
        self,
        hidden_size,
        conv_hidden_size,
        gate_pool_stride,
        gate_pool_kernel_size,
        gate_hidden_size,
        gate_conv_kernel_size,
        gate_coef,
        gru_gate_coef,
        observation_space,
        lower_level_load_path,
        num_conv_layers,
        kernel_size,
        stride,
        action_space,
        lower_level_config,
        **kwargs,
    ):
        self.gru_gate_coef = gru_gate_coef
        self.gate_coef = gate_coef
        self.conv_hidden_size = conv_hidden_size
        observation_space = Obs(**observation_space.spaces)
        recurrence.Recurrence.__init__(
            self,
            hidden_size=hidden_size,
            encoder_hidden_size=gate_hidden_size,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )
        abstract_recurrence.Recurrence.__init__(
            self,
            conv_hidden_size=conv_hidden_size,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.zeta = init_(nn.Linear(self.encoder_hidden_size, hidden_size))
        d, h, w = observation_space.obs.shape
        pool_input = int((h - kernel_size) / stride + 1)
        pool_output = int((pool_input - gate_pool_kernel_size) / gate_pool_stride + 1)
        self.gate_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=gate_pool_kernel_size, stride=gate_pool_stride),
            nn.Conv2d(
                in_channels=conv_hidden_size,
                out_channels=gate_hidden_size,
                kernel_size=min(pool_output, gate_conv_kernel_size),
                stride=2,
            ),
        )
        gc.collect()
        self.zeta2 = init_(
            nn.Linear(conv_hidden_size + self.encoder_hidden_size, hidden_size)
        )
        self.gru2 = LSTMCell(self.encoder_hidden_size, self.gru_hidden_size)
        self.d_gate = Categorical(hidden_size, 2)
        self.a_gate = Categorical(hidden_size, 2)
        state_sizes = self.state_sizes._asdict()
        with lower_level_config.open() as f:
            lower_level_params = json.load(f)
        self.state_sizes = RecurrentState(
            **state_sizes,
            hy=self.gru_hidden_size,
            cy=self.gru_hidden_size,
            ag_probs=2,
            dg_probs=2,
            ag=1,
            dg=1,
            gru_gate=self.gru_hidden_size,
            l=1,
            lh=lower_level_params["hidden_size"],
            P=self.ne * 2 * self.train_lines ** 2,
        )
        self.lower_level = None
        if lower_level_load_path is not None:
            ll_action_space = spaces.Discrete(Action(*action_space.nvec).lower)
            self.lower_level = Agent(
                obs_spaces=observation_space,
                entropy_coef=0,
                action_space=ll_action_space,
                lower_level=True,
                num_layers=1,
                **lower_level_params,
            )
            state_dict = torch.load(lower_level_load_path, map_location="cpu")
            self.lower_level.load_state_dict(state_dict["agent"])
            print(f"Loaded lower_level from {lower_level_load_path}.")

    @property
    def gru_in_size(self):
        return self.encoder_hidden_size

    def get_obs_sections(self, obs_spaces):
        try:
            obs_spaces = Obs(**obs_spaces)
        except TypeError:
            pass
        return super().get_obs_sections(obs_spaces)

    def set_obs_space(self, obs_space):
        super().set_obs_space(obs_space)
        self.obs_spaces = Obs(**self.obs_spaces)

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

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        state_sizes = self.state_sizes._replace(P=0)
        if hx.size(-1) == sum(self.state_sizes):
            state_sizes = self.state_sizes
        return RecurrentState(*torch.split(hx, state_sizes, dim=-1))

    def parse_input(self, x: torch.Tensor) -> ParsedInput:
        return ParsedInput(
            *torch.split(
                x,
                ParsedInput(obs=sum(self.obs_sections), actions=self.action_size),
                dim=-1,
            )
        )

    def inner_loop(self, raw_inputs, rnn_hxs):
        T, N, dim = raw_inputs.shape
        # raw_inputs, actions = torch.split(
        #     raw_inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        # )
        inputs = self.parse_input(raw_inputs)

        # parse non-action inputs
        state = Obs(*self.parse_obs(inputs.obs))
        state = state._replace(obs=state.obs.view(T, N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(T, N, *self.obs_spaces.lines.shape)

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        M = self.embed_task(self.preprocess_embed(N, T, state)).view(
            N, -1, self.encoder_hidden_size
        )

        P = self.build_P(M, N, rnn_hxs.device, nl)

        half = P.size(2) // 2 if self.no_scan else nl
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        hy = hx.hy
        cy = hx.cy
        p = hx.p.long().squeeze(-1)
        hx.a[new_episode] = self.n_a - 1
        ag_probs = hx.ag_probs
        ag_probs[new_episode, 1] = 1
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        actions = Action(*inputs.actions.unbind(dim=2))
        A = torch.cat([actions.upper, hx.a.view(1, N)], dim=0).long()
        L = torch.cat([actions.lower, hx.l.view(1, N) - 1], dim=0).long()
        D = torch.cat([actions.delta, hx.d.view(1, N)], dim=0).long()
        AG = torch.cat([actions.ag, hx.ag.view(1, N)], dim=0).long()
        DG = torch.cat([actions.dg, hx.dg.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            obs = self.conv(state.obs[t])
            # h = self.gru(obs, h)
            self.print("L[t]", L[t])
            self.print("lines[R, p]", lines[t][R, p])
            gate_obs = self.gate_conv(obs)
            # first put obs back in gru2
            z = F.relu(
                self.zeta2(
                    torch.cat(
                        [
                            M[R, p],
                            F.avg_pool2d(obs, kernel_size=obs.shape[-2:]).view(N, -1),
                        ],
                        dim=-1,
                    )
                )
            )
            # a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)

            # line_type, be, it, _ = lines[t][R, hx.p.long().flatten()].unbind(-1)
            # A[t] = 3 * (it - 1) + (be - 1)
            # print("*******")
            # print(be, it)
            # print(A[t])
            # print("*******")
            # A[:] = float(input("A:"))

            action = None if torch.any(L[t] < 0) else None
            ll_output = self.lower_level(
                Obs(**{k: v[t] for k, v in state._asdict().items()}),
                hx.lh,
                masks=None,
                action=action,
                upper=A[t],
            )
            L[t] = ll_output.action.flatten()
            embedded_lower = self.embed_lower(L[t].clone())

            z2 = F.relu(
                self.zeta(
                    (
                        M[R, p]
                        * F.max_pool2d(gate_obs, kernel_size=gate_obs.size(-1)).view(
                            N, -1
                        )
                        * embedded_lower
                    )
                )
            )
            # then put M back in gru
            # then put A back in gru
            d_gate = self.d_gate(z2)
            self.sample_new(DG[t], d_gate)
            # a_gate = self.a_gate(z)
            # self.sample_new(AG[t], a_gate)

            # (hy_, cy_), gru_gate = self.gru2(M[R, p], (hy, cy))
            u = self.upsilon(z).softmax(dim=-1)
            self.print("u", u)
            w = P[p, R]
            d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)
            dg = DG[t].unsqueeze(-1).float()
            self.print("dg prob", d_gate.probs[:, 1])
            self.print("dg", dg)
            d_dist = gate(dg, d_probs, ones * half)
            self.print("d_probs", d_probs[:, half:])
            self.sample_new(D[t], d_dist)
            # D[:] = float(input("D:")) + half
            p = p + D[t].clone() - half
            p = torch.clamp(p, min=0, max=M.size(1) - 1)

            # ag = AG[t].unsqueeze(-1).float()
            # a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            # self.sample_new(A[t], a_dist)
            # A[:] = float(input("A:"))
            # self.print("ag prob", a_gate.probs[:, 1])
            # self.print("ag", ag)
            # hy = dg * hy_ + (1 - dg) * hy
            # cy = dg * cy_ + (1 - dg) * cy
            yield RecurrentState(
                a=A[t],
                l=L[t],
                lh=hx.lh,
                v=self.critic(z),
                h=h,
                u=u,
                hy=hy,
                cy=cy,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=d_dist.probs,
                ag_probs=hx.ag_probs,
                dg_probs=d_gate.probs,
                ag=hx.ag,
                dg=dg,
                gru_gate=hx.gru_gate,
                P=P.transpose(0, 1),
            )
