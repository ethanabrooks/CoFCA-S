from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import NNBase, AgentValues
from ppo.control_flow.baselines import oh_et_al

from ppo.distributions import FixedCategorical, Categorical
from ppo.control_flow.env import Obs
from ppo.layers import Concat
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a p v h a_probs p_probs")


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


class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, entropy_coef, recurrent, baseline, a_equals_p, **network_args):
        nn.Module.__init__(self)
        self.a_equals_p = a_equals_p
        self.entropy_coef = entropy_coef
        if baseline == "oh-et-al":
            self.recurrent_module = oh_et_al.Recurrence(**network_args)
        else:
            self.recurrent_module = Recurrence(
                **network_args, baseline=baseline == "no-attention"
            )

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        rm = self.recurrent_module
        hx = RecurrentState(*rm.parse_hidden(all_hxs))
        a_dist = FixedCategorical(hx.a_probs)
        p_dist = FixedCategorical(hx.p_probs)
        if self.a_equals_p:
            action_log_probs = p_dist.log_probs(hx.p)
            entropy = p_dist.entropy().mean()
            action = torch.cat([hx.p, hx.p], dim=-1)
        else:
            action_log_probs = a_dist.log_probs(hx.a) + p_dist.log_probs(hx.p)
            entropy = (a_dist.entropy() + p_dist.entropy()).mean()
            action = torch.cat([hx.a, hx.p], dim=-1)
        return AgentValues(
            value=hx.v,
            action=action,
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=None,
            rnn_hxs=last_hx,
            log=dict(entropy=entropy),
        )

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
        else:
            y = torch.cat([x, action.float()], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    def get_value(self, inputs, rnn_hxs, masks):
        all_hxs, last_hx = self._forward_gru(
            inputs.view(inputs.size(0), -1), rnn_hxs, masks
        )
        return self.recurrent_module.parse_hidden(last_hx).v


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        activation,
        hidden_size,
        num_layers,
        debug,
        baseline,
    ):
        super().__init__()
        self.baseline = baseline
        self.obs_spaces = Obs(**observation_space.spaces)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.action_size = 2
        self.debug = debug
        self.hidden_size = hidden_size

        # networks
        nl = int(self.obs_spaces.lines.nvec[0])
        self.embeddings = nn.Embedding(nl, hidden_size)
        self.task_encoder = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        # f
        layers = [Concat(dim=-1)]
        in_size = self.obs_sections.condition + (2 if baseline else 1) * hidden_size
        for _ in range(num_layers + 1):
            layers.extend([nn.Linear(in_size, hidden_size), activation])
            in_size = hidden_size
        self.f = nn.Sequential(*layers)

        na = int(action_space.nvec[0])
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, na)
        self.linear = nn.Linear(hidden_size, 2 * hidden_size)
        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(na))
        self.state_sizes = RecurrentState(
            a=1, a_probs=na, p=1, p_probs=na, v=1, h=hidden_size
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
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)

        # build memory
        lines = inputs.lines.view(T, N, *self.obs_spaces.lines.shape).long()[0, :, :]
        M = self.embeddings(lines.view(-1)).view(
            *lines.shape, self.hidden_size
        )  # n_batch, n_lines, hidden_size
        forward_input = M.transpose(0, 1)  # n_lines, n_batch, hidden_size
        K, Kn = self.task_encoder(forward_input)
        Kn = Kn.transpose(0, 1).reshape(N, -1)
        K = K.transpose(0, 1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        p = hx.p.long()
        p[new_episode] = 0
        R = torch.arange(N, device=rnn_hxs.device)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        P = torch.cat([actions[:, :, 1], hx.p.view(1, N)], dim=0).long()

        for t in range(T):
            r = M[R, p.squeeze(1)]
            if self.baseline:
                h = self.gru(self.f((inputs.condition[t], Kn)), h)
            else:
                h = self.gru(self.f((inputs.condition[t], r)), h)
            a_dist = self.actor(h)
            q = self.linear(h)
            k = (K @ q.unsqueeze(2)).squeeze(2)
            self.print("k")
            self.print(k)
            p_dist = FixedCategorical(logits=k)
            self.print("dist")
            self.print(p_dist.probs)
            self.sample_new(P[t], p_dist)
            self.sample_new(A[t], a_dist)
            yield RecurrentState(
                a=A[t],
                v=self.critic(h),
                h=h,
                a_probs=a_dist.probs,
                p=P[t],
                p_probs=p_dist.probs,
            )
