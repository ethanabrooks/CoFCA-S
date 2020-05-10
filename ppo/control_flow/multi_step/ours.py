import gc
import json
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

import ppo.control_flow.multi_step.abstract_recurrence as abstract_recurrence
import ppo.control_flow.recurrence as recurrence
from ppo.agent import Agent
from ppo.layers import Flatten
from ppo.control_flow.env import Action
from ppo.control_flow.lstm import LSTMCell
from ppo.control_flow.multi_step.env import Obs
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState", "a l d u dg p v lh l_probs a_probs d_probs dg_probs P"
)

ParsedInput = namedtuple("ParsedInput", "obs actions")


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


def optimal_padding(kernel, stride):
    return (kernel // 2) % stride


def conv_output_dimension(h, padding, kernel, stride, dilation=1):
    return int(1 + (h + 2 * padding - dilation * (kernel - 1) - 1) / stride)


class Recurrence(abstract_recurrence.Recurrence, recurrence.Recurrence):
    def __init__(
        self,
        conv_hidden_size,
        conv2_hidden_size,
        kernel_size2,
        stride2,
        conv3_hidden_size,
        kernel_size3,
        stride3,
        kernel_size,
        stride,
        m_hidden_size,
        gate_coef,
        observation_space,
        lower_level_load_path,
        action_space,
        lower_level_config,
        embed_lower_hidden_size,
        **kwargs,
    ):
        self.gate_coef = gate_coef
        self.conv_hidden_size = conv_hidden_size
        observation_space = Obs(**observation_space.spaces)
        d, h, w = observation_space.obs.shape
        conv_h = conv_output_dimension(
            h=h,
            padding=optimal_padding(kernel_size, stride),
            kernel=kernel_size,
            stride=stride,
        )
        kernel_size2 = min(conv_h, kernel_size2)
        conv2_h = conv_output_dimension(
            h=conv_h,
            padding=optimal_padding(kernel_size2, stride2),
            kernel=kernel_size2,
            stride=stride2,
        )
        kernel_size3 = min(conv2_h, kernel_size3)
        conv3_h = conv_output_dimension(
            h=conv_h,
            padding=optimal_padding(kernel_size3, stride3),
            kernel=kernel_size3,
            stride=stride3,
        )
        # pool_output = int((conv_output - gate_pool_kernel_size) / gate_pool_stride + 1)

        conv2_output = conv3_h ** 2 * conv3_hidden_size

        recurrence.Recurrence.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=conv2_output,
            encoder_hidden_size=m_hidden_size,
            **kwargs,
        )
        abstract_recurrence.Recurrence.__init__(
            self,
            conv_hidden_size=conv_hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            encoder_hidden_size=m_hidden_size,
            num_conv_layers=1,
        )

        self.conv3_hidden_size = conv3_hidden_size
        self.conv2 = nn.Sequential(
            *(
                [
                    nn.Conv2d(
                        in_channels=conv_hidden_size + embed_lower_hidden_size,
                        out_channels=conv2_hidden_size,
                        kernel_size=kernel_size2,
                        stride=stride2,
                        padding=optimal_padding(kernel_size2, stride2),
                    )
                ]
                if conv_h > 1
                else [Flatten(), nn.Linear(conv_hidden_size, conv2_hidden_size)]
            ),
            nn.ReLU(),
            *(
                [
                    nn.Conv2d(
                        in_channels=conv2_hidden_size,
                        out_channels=conv3_hidden_size,
                        kernel_size=kernel_size3,
                        stride=stride3,
                        padding=optimal_padding(kernel_size3, stride3),
                    )
                ]
                if conv2_h > 1
                else [Flatten(), nn.Linear(conv2_hidden_size, conv3_hidden_size)]
            ),
        )
        self.embed_lower = nn.Embedding(
            self.action_space_nvec.lower + 1, embed_lower_hidden_size
        )
        self.actor = Categorical(conv_hidden_size + m_hidden_size, self.n_a)
        self.critic = init_(nn.Linear(conv3_hidden_size, 1))
        self.d_gate = Categorical(conv3_hidden_size, 2)
        self.upsilon = init_(nn.Linear(conv3_hidden_size, self.ne))
        state_sizes = self.state_sizes._asdict()
        with lower_level_config.open() as f:
            lower_level_params = json.load(f)
        ll_action_space = spaces.Discrete(Action(*action_space.nvec).lower)
        self.state_sizes = RecurrentState(
            **state_sizes,
            dg_probs=2,
            dg=1,
            l=1,
            l_probs=ll_action_space.n,
            lh=lower_level_params["hidden_size"],
            P=self.ne * 2 * self.train_lines ** 2,
        )
        self.lower_level = Agent(
            obs_spaces=observation_space,
            entropy_coef=0,
            action_space=ll_action_space,
            lower_level=True,
            num_layers=1,
            **lower_level_params,
        )
        if lower_level_load_path is not None:
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

        p = hx.p.long().squeeze(-1)
        hx.a[new_episode] = self.n_a - 1
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        actions = Action(*inputs.actions.unbind(dim=2))
        A = torch.cat([actions.upper, hx.a.view(1, N)], dim=0).long()
        L = torch.cat([actions.lower, hx.l.view(1, N) - 1], dim=0).long()
        D = torch.cat([actions.delta, hx.d.view(1, N)], dim=0).long()
        DG = torch.cat([actions.dg, hx.dg.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)

            obs = state.obs[t]
            conv_input = torch.cat(
                [
                    obs,
                    M[R, p].view(N, -1, 1, 1).expand(-1, -1, *obs.shape[-2:]),
                    state.inventory[t]
                    .view(N, -1, 1, 1)
                    .expand(-1, -1, *obs.shape[-2:]),
                ],
                dim=1,
            )
            conv_output = self.conv(conv_input).relu()
            a_dist = self.actor(
                torch.cat(
                    [conv_output.view(N, self.conv_hidden_size, -1).sum(-1), M[R, p]],
                    dim=-1,
                )
            )
            self.sample_new(A[t], a_dist)
            self.print("a_probs", a_dist.probs)
            # line_type, be, it, _ = lines[t][R, hx.p.long().flatten()].unbind(-1)
            # a = 3 * (it - 1) + (be - 1)
            # print("*******")
            # print(be, it)
            # print(A[t])
            # print("*******")

            ll_output = self.lower_level(
                Obs(**{k: v[t] for k, v in state._asdict().items()}),
                hx.lh,
                masks=None,
                action=None,
                upper=A[t],
            )
            if torch.any(L[0] < 0):
                assert torch.all(L[0] < 0)
                L[t] = ll_output.action.flatten()

            ac, be, it, _ = lines[t][R, p].long().unbind(-1)  # N, 2
            sell = (be == 2).long()
            channel_index = 3 * sell + (it - 1) * (1 - sell)
            channel = obs[R, channel_index]
            agent_channel = obs[R, -1]
            self.print("channel", channel)
            self.print("agent_channel", agent_channel)
            # not_subtask = (ac != 0).float().flatten()
            standing_on = (channel * agent_channel).view(N, -1).sum(-1)
            # correct_action = ((be - 1) == L[t]).float()
            # self.print("be", be)
            # self.print("L[t]", L[t])
            # self.print("correct_action", correct_action)
            # dg = standing_on * correct_action + not_subtask
            fuzz = (1 - standing_on).long() * torch.randint(
                2, size=(len(standing_on),), device=rnn_hxs.device
            )
            lt = (fuzz * (be - 1) + (1 - fuzz) * L[t]).long()
            # self.print("fuzz", fuzz, lt)
            # correct_action = ((be - 1) == lt).float()

            # h = self.gru(obs, h)
            embedded_lower = self.embed_lower(lt.clone())
            self.print("L[t]", L[t])
            self.print("lines[R, p]", lines[t][R, p])
            conv2_input = torch.cat(
                [
                    conv_output,
                    embedded_lower.view(N, -1, 1, 1).expand(
                        -1, -1, *conv_output.shape[-2:]
                    ),
                ],
                dim=1,
            )
            conv2_output = self.conv2(conv2_input)
            # then put M back in gru
            # then put A back in gru
            z = conv2_output.view(N, self.conv3_hidden_size, -1).sum(-1)
            d_gate = self.d_gate(z)
            self.sample_new(DG[t], d_gate)
            # (hy_, cy_), gru_gate = self.gru2(M[R, p], (hy, cy))
            # first put obs back in gru2

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
            # p = p + D[t].clone() - half
            p = p + dg.flatten().long()
            p = torch.clamp(p, min=0, max=M.size(1) - 1)

            # try:
            # A[:] = float(input("A:"))
            # except ValueError:
            # pass
            # hy = dg * hy_ + (1 - dg) * hy
            # cy = dg * cy_ + (1 - dg) * cy
            yield RecurrentState(
                a=A[t],
                l=L[t],
                lh=hx.lh,
                v=self.critic(z),
                u=u,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=d_dist.probs,
                dg_probs=d_gate.probs,
                l_probs=ll_output.dist.probs,
                dg=dg,
                P=P.transpose(0, 1),
            )
