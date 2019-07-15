from collections import namedtuple
from enum import Enum

from gym.spaces import MultiDiscrete
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ppo.control_flow
import ppo.control_flow.recurrence
from ppo.distributions import FixedCategorical
from ppo.layers import Flatten, Parallel, Product, Reshape, ShallowCopy, Sum
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState",
    "P a cg cr r p g a_probs cg_probs cr_probs g_probs v g_loss subtask",
)

L = Enum("Lines", "If Else Endif While EndWhile Subtask")


class Agent(ppo.control_flow.Agent):
    def load_agent(self, obs_spaces, **agent_args):
        return super().load_agent(
            obs_spaces=obs_spaces._replace(
                subtask=MultiDiscrete(obs_spaces.subtask.nvec[:3])
            ),
            **agent_args,
        )

    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def __init__(self, hidden_size, obs_spaces, **kwargs):
        self.original_obs_sections = [int(np.prod(s.shape)) for s in obs_spaces]
        super().__init__(hidden_size=hidden_size, obs_spaces=obs_spaces, **kwargs)
        d, h, w = self.obs_shape
        self.line_size = int(self.obs_spaces.subtasks.nvec[0].sum())
        h_size = d * self.line_size

        self.phi_shift = nn.Sequential(
            Parallel(
                nn.Sequential(Reshape(1, d, h, w)),
                nn.Sequential(Reshape(self.line_size, 1, 1, 1)),
            ),
            Product(),
            Reshape(d * self.line_size, *self.obs_shape[-2:]),
            # init_(
            # nn.Conv2d(self.condition_size * d, hidden_size, kernel_size=1, stride=1)
            # ),
            # attention {
            ShallowCopy(2),
            Parallel(
                Reshape(h_size, h * w),
                nn.Sequential(
                    init_(nn.Conv2d(h_size, 1, kernel_size=1)),
                    Reshape(1, h * w),
                    nn.Softmax(dim=-1),
                ),
            ),
            Product(),
            Sum(dim=-1),
            # }
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(h_size, 1), "sigmoid"),
            # init_(nn.Linear(d * self.condition_size * 4 * 4, 1), "sigmoid"),
            nn.Sigmoid(),
            Reshape(1, 1),
        )

        one_step = F.pad(torch.eye(self.n_subtasks - 1), [1, 0, 0, 1])
        one_step[:, -1] += 1 - one_step.sum(-1)
        self.register_buffer("one_step", one_step.unsqueeze(0))
        self.register_buffer(
            f"part3_one_hot", torch.eye(int(self.obs_spaces.subtasks.nvec[0, -1]))
        )
        no_op_probs = torch.zeros(1, self.actor.linear.out_features)
        no_op_probs[:, -1] = 1
        self.register_buffer("no_op_probs", no_op_probs)
        self.size_agent_subtask = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())
        self.agent_input_size = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())
        self.zeta = nn.Sequential(
            init_(nn.Linear(self.line_size, len(L))), nn.Softmax()
        )
        self.state_sizes = RecurrentState(
            **self.state_sizes._asdict(),
            P=self.obs_spaces.subtasks.nvec[0, -1]
            + 1,  # +1 for previous evaluation of condition
        )

    def get_a_dist(self, conv_out, g_binary, obs):
        probs = (
            super()
            .get_a_dist(conv_out, g_binary[:, : self.agent_input_size], obs)
            .probs
        )
        op = g_binary[:, self.agent_input_size].unsqueeze(1)
        no_op = 1 - op

        return FixedCategorical(
            # if subtask is a control-flow statement, force no-op
            probs=op * probs
            + no_op * self.no_op_probs.expand(op.size(0), -1)
        )

    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def inner_loop(self, M, inputs, gating_function, hx, **kwargs):
        i = self.obs_spaces.subtasks.nvec[0, -1]
        non_lines = torch.all(M == 0, dim=-1)
        M_zeta = self.zeta(M)

        def scan_forward():
            p = torch.zeros_like(hx.p)
            u = torch.cumsum(hx.p, dim=-1)
            for i in range(M.size(1)):
                p[:, i] = (
                    1
                    - p[:, :i].sum(-1) * M_zeta[L.EndIf, L.Else, L.EndWhile].sum(-1) * u
                )
            return p

        def scan_backward():
            p = torch.zeros_like(hx.p)
            u = torch.cumsum(hx.p.flip(-1), dim=-1).flip(-1)
            for i in range(M.size(1) - 1, -1, -1):
                p[:, i] = 1 - p[:, i:].sum(-1) * M_zeta[L.While] * u
            return p

        def update_attention(p, t):
            e = (p.unsqueeze(1) @ M_zeta).squeeze(1)
            eP = e[L.If, L.While].sum(-1, keepdim=True)
            er = eP / e[L.If, L.Else, L.While, L.EndWhile].sum(-1, keepdim=True)
            r = (p.unsqueeze(1) @ M).squeeze(1)
            r = er * r + (1 - er) * hx.P
            l = self.phi_shift((inputs.base[t], r))
            P = (
                e[L.If] * F.pad(l, [0, self.line_size])  # record evaluation
                + e[L.While] * F.pad(r, [1, 0])  # record condition
                + (1 - e[L.If] - e[L.While]) * hx.P  # keep the same
            )
            p_forward = scan_forward()
            p_backward = scan_backward()
            p_step = (p.unsqueeze(1) @ self.one_step).squeeze(1)
            return (
                e[L.If, L.While, L.Else].sum(-1) * (l * p_step + (1 - l) * p_forward)
                + e[L.EndWhile] * (l * p_backward + (1 - l) * p_step)
                + e[L.EndIf] * p_step
                + e[L.Subtask] * (c * p_step + (1 - c) * hx.p)
            )

        def _gating_function(subtask_param, **_kwargs):
            c, probs = gating_function(subtask_param, **_kwargs)
            return e[L.Subtask] * c + (1 - e[L.Subtask]), probs

        kwargs.update(update_attention=update_attention)
        is_subtask = M[:, :, -i].unsqueeze(-1)
        M[:, :, :-i] *= is_subtask
        for recurrent_state in super().inner_loop(
            gating_function=_gating_function, inputs=inputs, M=M, hx=hx, **kwargs
        ):
            yield RecurrentState(**recurrent_state._asdict(), P=P)
