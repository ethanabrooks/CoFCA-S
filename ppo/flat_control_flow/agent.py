import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from gridworld_env.flat_control_gridworld import Obs
import ppo.control_flow
from ppo.distributions import FixedCategorical
import ppo.subtasks
from ppo.layers import Parallel, Reshape, Product, ShallowCopy, Sum, Flatten
from ppo.utils import init_


class Agent(ppo.control_flow.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.control_flow.agent.Recurrence):
    def __init__(self, hidden_size, obs_spaces, **kwargs):
        self.original_obs_sections = [int(np.prod(s.shape)) for s in obs_spaces]
        super().__init__(
            hidden_size=hidden_size,
            obs_spaces=obs_spaces._replace(subtasks=obs_spaces.lines),
            **kwargs,
        )
        true_path = F.pad(torch.eye(self.n_subtasks - 1), [1, 0, 0, 1])
        true_path[:, -1] += 1 - true_path.sum(-1)
        self.register_buffer("true_path", true_path)
        false_path = F.pad(torch.eye(self.n_subtasks - 2), [2, 0, 0, 2])
        false_path[:, -1] += 1 - false_path.sum(-1)
        self.register_buffer("false_path", false_path)
        self.register_buffer(
            f"part3_one_hot", torch.eye(int(self.obs_spaces.lines.nvec[0, -1]))
        )
        no_op_probs = torch.zeros(1, self.actor.linear.out_features)
        no_op_probs[:, -1] = 1
        self.register_buffer("no_op_probs", no_op_probs)
        self.size_agent_subtask = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())

    def parse_inputs(self, inputs):
        obs = Obs(*torch.split(inputs, self.original_obs_sections, dim=2))
        return obs._replace(subtasks=obs.lines)

    def get_a_dist(self, conv_out, g_binary, obs):
        probs = (
            super().get_a_dist(conv_out, g_binary[:, : -self.condition_size], obs).probs
        )
        op = g_binary[:, -self.condition_size].unsqueeze(1)
        no_op = 1 - op

        return FixedCategorical(
            # if subtask is a control-flow statement, force no-op
            probs=op * probs
            + no_op * self.no_op_probs.expand(op.size(0), -1)
        )

    def build_phi_shift(self, d, h, hidden_size, w):
        return nn.Sequential(
            Parallel(
                nn.Sequential(Reshape(1, d, h, w)),
                nn.Sequential(Reshape(self.condition_size, 1, 1, 1)),
            ),
            Product(),
            # Reshape(d * self.condition_size, *self.obs_shape[-2:]),
            # init_(
            #     nn.Conv2d(self.condition_size * d, hidden_size, kernel_size=1, stride=1)
            # ),
            # # attention {
            # ShallowCopy(2),
            # Parallel(
            #     Reshape(hidden_size, h * w),
            #     nn.Sequential(
            #         init_(nn.Conv2d(hidden_size, 1, kernel_size=1)),
            #         Reshape(1, h * w),
            #         nn.Softmax(dim=-1),
            #     ),
            # ),
            # Product(),
            # Sum(dim=-1),
            # # }
            # nn.ReLU(),
            # Flatten(),
            init_(nn.Linear(1 + h * w * (self.condition_size - 1), 1), "sigmoid"),
            nn.Sigmoid(),
            Reshape(1, 1),
        )

    def inner_loop(self, M, inputs, **kwargs):
        def update_attention(p, t):
            # r = (p.unsqueeze(1) @ M).squeeze(1)
            r = (p.unsqueeze(1) @ M).squeeze(1)
            obs = inputs.base[t, :, 1:-2]
            N = p.size(0)
            i = self.obs_spaces.subtasks.nvec[0, -1]
            condition = r[:, -i:].view(N, i, 1, 1)
            is_subtask = condition[:, 0].view(N, -1)
            pred = ((condition[:, 1:] * obs) > 0).view(N, -1).float()
            # pred = ((condition[:, 1:] * obs) > 0).view(N, 1, 1, -1).any(dim=-1).float()
            # pred = self.phi_shift((inputs.base[t], r[:, -self.condition_size :]))
            # take_two_steps = (1 - is_subtask) * (1 - pred)
            # truth = 1 - take_two_steps
            phi_in = torch.cat([is_subtask, pred], dim=-1)
            pred = self.phi_shift(phi_in)  # TODO
            trans = pred * self.true_path + (1 - pred) * self.false_path
            x = (p.unsqueeze(1) @ trans).squeeze(1)
            # if torch.any(x < 0):
            # import ipdb

            # ipdb.set_trace()
            return x

        kwargs.update(update_attention=update_attention)
        yield from ppo.subtasks.Recurrence.inner_loop(
            self, inputs=inputs, M=M, **kwargs
        )
