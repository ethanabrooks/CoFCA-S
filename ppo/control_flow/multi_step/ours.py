import json
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from gym import spaces

import ppo.control_flow.multi_step.abstract_recurrence as abstract_recurrence
import ppo.control_flow.recurrence as recurrence
from ppo.agent import Agent
from ppo.control_flow.env import Action
from ppo.control_flow.multi_step.env import Obs
from ppo.distributions import FixedCategorical, Categorical
from ppo.layers import Flatten
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
        action_space,
        concat,
        conv_hidden_size,
        gate_coef,
        observation_space,
        lower_level_load_path,
        action_space,
        lower_level_config,
        embed_lower_hidden_size,
        **kwargs,
    ):
        self.gate_coef = gate_coef
        self.gate_hidden_size = gate_hidden_size
        self.gate_kernel_size = gate_kernel_size
        self.gate_stride = gate_stride
        self.conv_hidden_size = conv_hidden_size
        self.stride = stride
        observation_space = Obs(**observation_space.spaces)

        recurrence.Recurrence.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            encoder_hidden_size=m_hidden_size,
            **kwargs,
        )
        abstract_recurrence.Recurrence.__init__(
            self,
            conv_hidden_size=conv_hidden_size,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.embed_lower = nn.Embedding(
            self.action_space_nvec.lower + 1,
            # encoder_hidden_size
            gate_hidden_size,
        )

        def generate_convolutions(d, h, w):
            in_size = d + embed_lower_hidden_size

            for hidden, kernel, stride in conv_params:
                if hidden <= 0:
                    break
                kernel1 = min(h, kernel)
                kernel2 = min(w, kernel)
                stride1 = min(kernel1, stride)
                stride2 = min(kernel2, stride)
                padding1 = optimal_padding(kernel1, stride1)
                padding2 = optimal_padding(kernel2, stride2)
                if h > 1 or w > 1:
                    yield (
                        nn.Conv2d(
                            in_channels=in_size,
                            out_channels=hidden,
                            kernel_size=(kernel1, kernel2),
                            stride=(stride1, stride2),
                            padding=(padding1, padding2),
                        )
                    )
                    h = conv_output_dimension(
                        h, padding=padding1, kernel=kernel1, stride=stride1
                    )
                    w = conv_output_dimension(
                        w, padding=padding2, kernel=kernel2, stride=stride2
                    )
                else:
                    yield Flatten()
                    yield nn.Linear(in_size, hidden)
                yield nn.ReLU()
                in_size = hidden

        _, H, W, = self.obs_spaces.obs.shape
        kernel0 = min(H, kernel0)
        padding0 = optimal_padding(kernel0, stride0)

        self.conv2 = nn.Sequential(
            *generate_convolutions(
                hidden0,
                h=conv_output_dimension(
                    H, padding=padding0, kernel=kernel0, stride=stride0
                ),
                w=conv_output_dimension(
                    W, padding=padding0, kernel=kernel0, stride=stride0
                ),
            )
        self.gate_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=gate_pool_kernel_size, stride=gate_pool_stride),
            nn.Conv2d(
                in_channels=self.conv_hidden_size,
                out_channels=gate_conv_hidden_size
                if self.concat_gate
                else gate_hidden_size,
                kernel_size=min(pool_output, gate_conv_kernel_size),
                stride=2,
            ),
        )
        inventory_size = self.obs_spaces.inventory.n
        self.embed_inventory = nn.Sequential(
            init_(nn.Linear(inventory_size, inventory_hidden_size)), nn.ReLU()
        )
        self.zeta = init_(
            nn.Linear(
                self.conv_hidden_size + self.encoder_hidden_size + inventory_hidden_size
                if concat
                else hidden_size,
                hidden_size,
            )
        )

        output_dim = conv_output_dimension(
            h=h,
            padding=optimal_padding(gate_kernel_size, gate_stride),
            kernel=gate_kernel_size,
            stride=gate_stride,
        )
        output_dim2 = conv_output_dimension(
            h=output_dim,
            padding=optimal_padding(kernel_size, stride),
            kernel=kernel_size,
            stride=stride,
        )
        output_dim3 = conv_output_dimension(
            h=output_dim2,
            padding=optimal_padding(kernel_size, stride),
            kernel=kernel_size,
            stride=stride,
        )
        self.d_gate = Categorical(
            self.encoder_hidden_size
            + hidden2
            + self.conv_hidden_size * output_dim3 ** 2,
            2,
        )
        self.linear1 = nn.Linear(
            self.encoder_hidden_size,
            self.conv_hidden_size * kernel_size ** 2 * self.conv_hidden_size,
        )

        self.conv_bias = nn.Parameter(torch.zeros(self.conv_hidden_size))
        self.linear2 = nn.Linear(self.encoder_hidden_size + gate_hidden_size, hidden2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.conv_hidden_size,
                kernel_size=kernel,
                padding=padding,
                stride=2,
            )
        )
        self.linear2 = nn.Linear(self.task_embed_size + lower_embed_size, hidden2)
        self.conv_bias = nn.Parameter(torch.zeros(gate_hidden_size))
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
        return self.task_embed_size

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
            N, -1, self.task_embed_size
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
            conv_output = self.conv(state.obs[t]).sum(-1).sum(-1)
            inventory = self.embed_inventory(state.inventory[t])
            zeta_input = (
                torch.cat([m, obs_conv_output, inventory], dim=-1)
                if self.concat
                else (m * obs_conv_output * inventory)
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

            # ac, be, it, _ = lines[t][R, p].long().unbind(-1)  # N, 2
            # sell = (be == 2).long()
            # channel_index = 3 * sell + (it - 1) * (1 - sell)
            # channel = state.obs[t][R, channel_index]
            # agent_channel = state.obs[t][R, -1]
            # self.print("channel", channel)
            # self.print("agent_channel", agent_channel)
            # not_subtask = (ac != 0).float().flatten()
            # standing_on = (channel * agent_channel).view(N, -1).sum(-1)
            # correct_action = ((be - 1) == L[t]).float()
            # self.print("be", be)
            # self.print("L[t]", L[t])
            # self.print("correct_action", correct_action)
            # dg = standing_on * correct_action + not_subtask
            # fuzz = (1 - dg).long() * torch.randint(
            #     2, size=(len(dg),), device=rnn_hxs.device
            # )
            # lt = (fuzz * (be - 1) + (1 - fuzz) * L[t]).long()
            # self.print("fuzz", fuzz, lt)
            # dg = dg.view(N, 1)
            # correct_action = ((be - 1) == lt).float()

            # h = self.gru(obs, h)
            embedded_lower = self.embed_lower(L[t].clone())
            self.print("L[t]", L[t])
            self.print("lines[R, p]", lines[t][R, p])
            # then put M back in gru
            # then put A back in gru
            conv_kernel = self.linear1(M[R, p]).view(
                N,
                self.gate_hidden_size,
                -1,
                self.gate_kernel_size,
                self.gate_kernel_size,
            )
            padding = optimal_padding(self.kernel_size, self.stride)
            h1 = torch.cat(
                [
                    F.conv2d(
                        input=o.unsqueeze(0),
                        weight=k,
                        bias=self.conv_bias,
                        stride=self.gate_stride,
                        padding=padding,
                    )
                    for o, k in zip(obs.relu().unbind(0), conv_kernel.unbind(0))
                ],
                dim=0,
            ).relu()
            h2 = self.linear2(
                torch.cat([M[R, p], embedded_lower], dim=-1)
            ).relu()  # TODO: use bilinear here?
            d_gate = self.d_gate(
                torch.cat([h1.view(N, -1), h2, M[R, p]], dim=-1)
            )  # TODO: remove M[R, p] here?
            self.sample_new(DG[t], d_gate)

            u = self.upsilon(z).softmax(dim=-1)
            # TODO: feed h1 instead of z?
            self.print("u", u)
            w = P[p, R]
            d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)
            dg = DG[t].unsqueeze(-1).float()
            # correct_action = self.linear(z).sigmoid()

            self.print("dg prob", d_gate.probs[:, 1])
            self.print("dg", dg)
            d_dist = gate(dg, d_probs, ones * half)
            self.print("d_probs", d_probs[:, half:])
            self.sample_new(D[t], d_dist)
            # D[:] = float(input("D:")) + half
            p = p + D[t].clone() - half
            # p = p + dg.flatten().long()
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
