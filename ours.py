import json
from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from distributions import FixedCategorical, Categorical
from env import Action
from lower_level import Agent, get_obs_sections
from env import Obs
from transformer import TransformerModel
from utils import init_

RecurrentState = namedtuple(
    "RecurrentState", "a l d h dg p v lh l_probs a_probs d_probs dg_probs"
)

ParsedInput = namedtuple("ParsedInput", "obs actions")


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


def optimal_padding(h, kernel, stride):
    n = np.ceil((h - kernel) / stride + 1)
    return int(np.ceil((stride * (n - 1) + kernel - h) / 2))


def conv_output_dimension(h, padding, kernel, stride, dilation=1):
    return int(1 + (h + 2 * padding - dilation * (kernel - 1) - 1) / stride)


class Recurrence(nn.Module):
    def __init__(
        self,
        hidden_size,
        conv_hidden_size,
        fuzz,
        inventory_hidden_size,
        gate_coef,
        observation_space,
        lower_level_load_path,
        lower_embed_size,
        kernel_size,
        stride,
        action_space,
        lower_level_config,
        task_embed_size,
        num_edges,
        activation,
        num_layers,
        olsk,
        no_pointer,
        transformer,
        log_dir,
        no_roll,
        no_scan,
        debug,
        eval_lines,
    ):
        super().__init__()
        self.fuzz = fuzz
        self.gate_coef = gate_coef
        self.conv_hidden_size = conv_hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        observation_space = Obs(**observation_space.spaces)
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

        self.obs_sections = get_obs_sections(self.obs_spaces)
        self.eval_lines = eval_lines
        self.train_lines = len(self.obs_spaces.lines.nvec)

        # networks
        if olsk:
            num_edges = 3
        self.ne = num_edges
        self.action_space_nvec = Action(*map(int, action_space.nvec))
        self.n_a = n_a = self.action_space_nvec.upper
        self.embed_task = nn.EmbeddingBag(
            self.obs_spaces.lines.nvec[0].sum(), task_embed_size
        )
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

        self.actor = Categorical(hidden_size, n_a)
        self.conv_hidden_size = conv_hidden_size
        d, h, _ = self.obs_spaces.obs.shape
        self.register_buffer("ones", torch.ones(1, dtype=torch.long))
        self.register_buffer(
            "offset",
            F.pad(torch.tensor(self.obs_spaces.lines.nvec[0, :-1]).cumsum(0), [1, 0]),
        )
        d, h, w = observation_space.obs.shape
        self.obs_dim = d
        self.kernel_size = min(d, kernel_size)
        self.padding = padding = optimal_padding(h, kernel_size, stride) + 1
        self.embed_lower = nn.Embedding(
            self.action_space_nvec.lower + 1, lower_embed_size
        )
        self.embed_inventory = nn.Sequential(
            init_(nn.Linear(self.obs_spaces.inventory.n, inventory_hidden_size)),
            nn.ReLU(),
        )
        m_size = (
            2 * self.task_embed_size + hidden_size
            if self.no_pointer
            else self.task_embed_size
        )
        zeta1_input_size = m_size + self.conv_hidden_size + inventory_hidden_size
        self.zeta1 = init_(nn.Linear(zeta1_input_size, hidden_size))
        z2_size = zeta1_input_size + lower_embed_size
        if self.olsk:
            assert self.ne == 3
            self.upsilon = nn.GRUCell(z2_size, hidden_size)
            self.beta = init_(nn.Linear(hidden_size, self.ne))
        elif self.no_pointer:
            self.upsilon = nn.GRUCell(z2_size, hidden_size)
            self.beta = init_(nn.Linear(hidden_size, self.d_space()))
        else:
            self.upsilon = init_(nn.Linear(z2_size, self.ne))
            in_size = (2 if self.no_roll or self.no_scan else 1) * task_embed_size
            out_size = self.ne * self.d_space() if self.no_scan else self.ne
            self.beta = nn.Sequential(init_(nn.Linear(in_size, out_size)))
        self.d_gate = Categorical(z2_size, 2)
        self.kernel_net = nn.Linear(m_size, conv_hidden_size * kernel_size ** 2 * d)
        self.conv_bias = nn.Parameter(torch.zeros(conv_hidden_size))
        self.critic = init_(nn.Linear(hidden_size, 1))
        with lower_level_config.open() as f:
            lower_level_params = json.load(f)
        ll_action_space = spaces.Discrete(Action(*action_space.nvec).lower)
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=n_a,
            d=1,
            d_probs=(self.d_space()),
            h=hidden_size,
            p=1,
            v=1,
            dg_probs=2,
            dg=1,
            l=1,
            l_probs=ll_action_space.n,
            lh=lower_level_params["hidden_size"],
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

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        state_sizes = self.state_sizes
        # noinspection PyArgumentList
        if hx.size(-1) == sum(self.state_sizes):
            state_sizes = self.state_sizes
        return RecurrentState(*torch.split(hx, state_sizes, dim=-1))

    # noinspection PyPep8Naming
    def inner_loop(self, raw_inputs, rnn_hxs):
        T, N, dim = raw_inputs.shape
        inputs = ParsedInput(
            *torch.split(
                raw_inputs,
                ParsedInput(obs=sum(self.obs_sections), actions=self.action_size),
                dim=-1,
            )
        )

        # parse non-action inputs
        state = Obs(*torch.split(inputs.obs, self.obs_sections, dim=-1))
        state = state._replace(obs=state.obs.view(T, N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(T, N, *self.obs_spaces.lines.shape)

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        M = self.embed_task(
            (
                state.lines.view(T, N, *self.obs_spaces.lines.shape).long()[0, :, :]
                + self.offset
            ).view(-1, self.obs_spaces.lines.nvec[0].size)
        ).view(N, -1, self.task_embed_size)
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

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
            P = self.task_encoder(M.transpose(0, 1)).view(nl, N, -1, self.ne).softmax(2)
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
            B = self.beta(G).sigmoid()
            # arange = torch.zeros(6).float()
            # arange[0] = 1
            # arange[1] = 1
            # B[:, :, :, 0] = 0  # arange.view(1, 1, -1, 1)
            # B[:, :, :, 1] = 1
            f, b = torch.unbind(B, dim=3)
            B = torch.stack([f, b.flip(2)], dim=-2)
            B = B.view(nl, N, 2 * nl, self.ne)
            last = torch.zeros(nl, N, 2 * nl, self.ne, device=rnn_hxs.device)
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
            # noinspection PyArgumentList
            half = P.size(2) // 2 if self.no_scan else nl

        p = hx.p.long().squeeze(-1)
        h = hx.h
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
            m = torch.cat([P, h], dim=-1) if self.no_pointer else M[R, p]
            conv_kernel = self.kernel_net(m).view(
                N,
                self.conv_hidden_size,
                self.obs_dim,
                self.kernel_size,
                self.kernel_size,
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
                    for o, k in zip(state.obs[t].unbind(0), conv_kernel.unbind(0))
                ],
                dim=0,
            ).relu()
            h1 = h1.sum(-1).sum(-1)
            inventory = self.embed_inventory(state.inventory[t])
            zeta1_input = torch.cat([m, h1, inventory], dim=-1)
            z1 = F.relu(self.zeta1(zeta1_input))
            a_dist = self.actor(z1)
            self.sample_new(A[t], a_dist)
            a = A[t]
            self.print("a_probs", a_dist.probs)
            # line_type, be, it, _ = lines[t][R, hx.p.long().flatten()].unbind(-1)
            # a = 3 * (it - 1) + (be - 1)

            ll_output = self.lower_level(
                Obs(**{k: v[t] for k, v in state._asdict().items()}),
                hx.lh,
                masks=None,
                action=None,
                upper=a,
            )
            if torch.any(L[0] < 0):
                assert torch.all(L[0] < 0)
                L[t] = ll_output.action.flatten()

            if self.fuzz:
                ac, be, it, _ = lines[t][R, p].long().unbind(-1)  # N, 2
                sell = (be == 2).long()
                channel_index = 3 * sell + (it - 1) * (1 - sell)
                channel = state.obs[t][R, channel_index]
                agent_channel = state.obs[t][R, -1]
                # self.print("channel", channel)
                # self.print("agent_channel", agent_channel)
                is_subtask = (ac == 0).flatten()
                standing_on = (channel * agent_channel).view(N, -1).sum(-1)
                # correct_action = ((be - 1) == L[t]).float()
                # self.print("be", be)
                # self.print("L[t]", L[t])
                # self.print("correct_action", correct_action)
                # dg = standing_on * correct_action + not_subtask
                fuzz = (
                    is_subtask.long()
                    * (1 - standing_on).long()
                    * torch.randint(2, size=(len(standing_on),), device=rnn_hxs.device)
                )
                lt = (fuzz * (be - 1) + (1 - fuzz) * L[t]).long()
                self.print("fuzz", fuzz, lt)
                # dg = dg.view(N, 1)
                # correct_action = ((be - 1) == lt).float()
            else:
                lt = L[t]

            embedded_lower = self.embed_lower(lt.clone())
            self.print("L[t]", L[t])
            self.print("lines[R, p]", lines[t][R, p])
            z2 = torch.cat([zeta1_input, embedded_lower], dim=-1)
            d_gate = self.d_gate(z2)
            self.sample_new(DG[t], d_gate)
            dg = DG[t].unsqueeze(-1).float()

            # _, _, it, _ = lines[t][R, p].long().unbind(-1)  # N, 2
            # sell = (be == 2).long()
            # index1 = it - 1
            # index2 = 1 + ((it - 3) % 3)
            # channel1 = state.obs[t][R, index1].sum(-1).sum(-1)
            # channel2 = state.obs[t][R, index2].sum(-1).sum(-1)
            # z = (channel1 > channel2).unsqueeze(-1).float()

            if self.olsk or self.no_pointer:
                h = self.upsilon(z2, h)
                u = self.beta(h).softmax(dim=-1)
                d_dist = gate(dg, u, ones)
                self.sample_new(D[t], d_dist)
                delta = D[t].clone() - 1
            else:
                u = self.upsilon(z2).softmax(dim=-1)
                self.print("u", u)
                w = P[p, R]
                d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)

                self.print("dg prob", d_gate.probs[:, 1])
                self.print("dg", dg)
                d_dist = gate(dg, d_probs, ones * half)
                self.print("d_probs", d_probs[:, half:])
                self.sample_new(D[t], d_dist)
                # D[:] = float(input("D:")) + half
                delta = D[t].clone() - half
                self.print("D[t], delta", D[t], delta)
                P.view(N, *self.P_shape())
            p = p + delta
            p = torch.clamp(p, min=0, max=M.size(1) - 1)

            # try:
            # A[:] = float(input("A:"))
            # except ValueError:
            # pass
            yield RecurrentState(
                a=A[t],
                l=L[t].detach(),
                lh=hx.lh,
                v=self.critic(z1),
                h=h,
                p=p,
                d=D[t],
                dg=dg,
                a_probs=a_dist.probs,
                d_probs=d_dist.probs,
                dg_probs=d_gate.probs,
                l_probs=ll_output.dist.probs,
            )

    @property
    def gru_in_size(self):
        return self.hidden_size + self.conv_hidden_size + self.encoder_hidden_size

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

    # noinspection PyProtectedMember
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
            for name, size, hx in zip(
                RecurrentState._fields, self.state_sizes, zip(*hxs)
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
