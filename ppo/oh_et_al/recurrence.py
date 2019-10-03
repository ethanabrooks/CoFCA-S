from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import ppo.oh_et_al
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a a_probs b b_probs v0 v1 h0 h1 p")


def batch_conv1d(inputs, weights):
    outputs = []
    # one convolution per instance
    n = inputs.shape[0]
    for i in range(n):
        x = inputs[i]
        w = weights[i]
        convolved = F.conv1d(x.reshape(1, 1, -1), w.reshape(1, 1, -1), padding=2)
        outputs.append(convolved.squeeze(0))
    padded = torch.cat(outputs)
    padded[:, 1] = padded[:, 1] + padded[:, 0]
    padded[:, -2] = padded[:, -2] + padded[:, -1]
    return padded[:, 1:-1]


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        activation,
        hidden_size,
        num_layers,
        debug,
    ):
        super().__init__()
        self.obs_spaces = ppo.oh_et_al.Obs(**observation_space.spaces)
        self.act_spaces = ppo.oh_et_al.Actions(**action_space.spaces)
        self.obs_sections = ppo.oh_et_al.Obs(
            *[int(np.prod(s.shape)) for s in self.obs_spaces]
        )
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        self.subtask_embeddings = nn.Sequential(
            nn.Embedding(int(self.obs_spaces.subtasks.nvec.max()), hidden_size),
            activation,
        )
        self.obs_embeddings = nn.Sequential(
            nn.Embedding(int(self.obs_spaces.obs.nvec.max()), hidden_size), activation
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2 * hidden_size, 2 * hidden_size, kernel_size=1), activation
        )
        self.gru0 = nn.GRU(
            hidden_size * (self.obs_sections.obs + self.obs_spaces.subtasks.shape[1]),
            hidden_size,
            num_layers,
        )
        self.gru1 = nn.GRU(
            hidden_size * (self.obs_sections.obs + self.obs_spaces.subtasks.shape[1]),
            hidden_size,
            num_layers,
        )
        self.critic0 = init_(nn.Linear(hidden_size, 1))
        self.critic1 = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, self.act_spaces.action.n)
        self.phi_update = Categorical(hidden_size, 2)
        self.state_sizes = RecurrentState(
            a=1,
            a_probs=self.act_spaces.action.n,
            b=1,
            b_probs=self.act_spaces.beta.n,
            p=1,
            v0=1,
            v1=1,
            h0=num_layers * hidden_size,
            h1=num_layers * hidden_size,
        )

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, hx):
        return self.pack(self.inner_loop(inputs, rnn_hxs=hx))

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

    def parse_inputs(self, inputs: torch.Tensor):
        return ppo.oh_et_al.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            torch.set_printoptions(precision=2, sci_mode=False)
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        device = inputs.device
        inputs, actions = torch.split(
            inputs.detach().long(), [D - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)

        # build memory
        M = self.subtask_embeddings(inputs.subtasks[0]).view(
            N, self.obs_spaces.subtasks.shape[0], -1
        )
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h0 = (
            hx.h0.view(N, self.gru0.num_layers, self.gru0.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        h1 = (
            hx.h1.view(N, self.gru0.num_layers, self.gru0.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        p = hx.p.long().squeeze(1)
        p[new_episode] = 0
        A = actions[:, :, 0]
        B = actions[:, :, 1]
        d, *dims = self.obs_spaces.obs.shape
        R = torch.arange(N, device=device)

        for t in range(T):
            r = M[R, p]
            obs = inputs.obs[t].view(N, d, *dims)
            embedded = self.obs_embeddings(obs)
            conv_in = embedded.permute(0, 1, 4, 2, 3).reshape(N, -1, *dims)
            conv_out = self.conv(conv_in)
            gru_inputs = torch.cat([conv_out.view(N, -1), r], dim=-1).unsqueeze(0)
            hn0, h0 = self.gru0(gru_inputs, h0)
            hn1, h1 = self.gru1(gru_inputs, h1)
            hn0 = hn0.squeeze(0)
            hn1 = hn1.squeeze(0)
            b_dist = self.phi_update(hn0)
            a_dist = self.actor(hn1)
            self.sample_new(A[t], a_dist)
            self.sample_new(B[t], b_dist)
            v0 = self.critic0(hn0)
            v1 = self.critic1(hn1)
            p = torch.min(p + B[t], (inputs.n_subtasks[t] - 1).flatten())
            self.print("p", p)
            self.print("v0", v0)
            self.print("v1", v1)
            h1 = h1 * B[t].unsqueeze(0).unsqueeze(-1).float()
            yield RecurrentState(
                a=A[t],
                b=B[t],
                a_probs=a_dist.probs,
                b_probs=b_dist.probs,
                v0=v0,
                v1=v1,
                h0=h0.transpose(0, 1),
                h1=h1.transpose(0, 1),
                p=p,
            )
