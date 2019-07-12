import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld_env.flat_control_gridworld import Obs
import ppo.control_flow
from ppo.distributions import FixedCategorical
from ppo.layers import Flatten, Parallel, Product, Reshape, ShallowCopy, Sum
import ppo.subtasks
from ppo.utils import init_


class Agent(ppo.control_flow.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.control_flow.agent.Recurrence):
    def __init__(self, hidden_size, obs_spaces, **kwargs):
        self.obs_sections = [int(np.prod(s.shape)) for s in obs_spaces]
        super().__init__(hidden_size=hidden_size, obs_spaces=obs_spaces, **kwargs)
        one_step = F.pad(torch.eye(self.n_subtasks - 1), [1, 0, 0, 1])
        one_step[:, -1] += 1 - one_step.sum(-1)
        self.register_buffer("one_step", one_step.unsqueeze(0))
        two_steps = F.pad(torch.eye(self.n_subtasks - 2), [2, 0, 0, 2])
        two_steps[:, -1] += 1 - two_steps.sum(-1)
        self.register_buffer("two_steps", two_steps.unsqueeze(0))
        self.register_buffer(
            f"part3_one_hot", torch.eye(int(self.obs_spaces.subtasks.nvec[0, -1]))
        )
        no_op_probs = torch.zeros(1, self.actor.linear.out_features)
        no_op_probs[:, -1] = 1
        self.register_buffer("no_op_probs", no_op_probs)
        self.size_agent_subtask = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())
        self.f = nn.Sequential(
            init_(nn.Linear(self.condition_size, 1), "sigmoid"),
            Reshape(1),
            nn.Sigmoid(),
        )
        self.agent_input_size = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())

    def parse_inputs(self, inputs):
        return Obs(*torch.split(inputs, self.obs_sections, dim=2))

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

    @property
    def condition_size(self):
        return int(self.obs_spaces.subtasks.nvec[0].sum())

    def inner_loop(self, M, inputs, gating_function, **kwargs):
        i = self.obs_spaces.subtasks.nvec[0, -1]

        def update_attention(p, t):
            # r = (p.unsqueeze(1) @ M).squeeze(1)
            r = (p.unsqueeze(1) @ M).squeeze(1)
            # N = p.size(0)
            # condition = r[:, -i:].view(N, i, 1, 1)
            # obs = inputs.base[t, :, 1:-2]
            # is_subtask = condition[:, 0]
            is_control_flow = self.f(r).unsqueeze(-1)
            is_subtask = 1 - is_control_flow
            # pred = ((condition[:, 1:] * obs) > 0).view(N, 1, 1, -1).any(dim=-1).float()
            condition_passes = self.phi_shift((inputs.base[t], r))
            condition_fails = 1 - condition_passes
            trans = condition_passes * self.one_step + condition_fails * self.two_steps
            trans = is_subtask * self.one_step + is_control_flow * trans
            return (p.unsqueeze(1) @ trans).squeeze(1)

        def _gating_function(subtask_param, **_kwargs):
            c, probs = gating_function(subtask_param, **_kwargs)
            is_control_flow = self.f(subtask_param)
            return c + is_control_flow - c * is_control_flow, probs

        kwargs.update(update_attention=update_attention)
        is_subtask = M[:, :, -i].unsqueeze(-1)
        M[:, :, :-i] *= is_subtask
        yield from ppo.subtasks.Recurrence.inner_loop(
            self, gating_function=_gating_function, inputs=inputs, M=M, **kwargs
        )
