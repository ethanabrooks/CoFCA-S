import numpy as np
import torch
import torch.nn as nn

import ppo.subtasks.agent
from gridworld_env.control_flow_gridworld import Branch
from ppo.control_flow.wrappers import Obs
from ppo.layers import Reshape, Flatten
from ppo.subtasks.teacher import g123_to_binary
from ppo.subtasks.wrappers import Actions


class Agent(ppo.subtasks.agent.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.subtasks.agent.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.obs_sections = Obs(
            *[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.register_buffer('branch_one_hots', torch.eye(2 * self.n_subtasks))
        num_object_types = self.obs_spaces.subtask.nvec[2]
        self.register_buffer('condition_one_hots', torch.eye(num_object_types))

        in_channels = num_object_types * self.obs_shape[0]
        self.phi_shift = nn.Sequential(
            Reshape(-1, in_channels, *self.obs_shape[-2:]),
            nn.Conv2d(in_channels, hidden_size, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=1, stride=1),
            Flatten(),
        )

    def forward(self, inputs, hx):
        assert hx is not None
        T, N, D = inputs.shape

        # detach actions
        # noinspection PyProtectedMember
        n_actions = len(Actions._fields)
        inputs, *actions = torch.split(
            inputs.detach(), [D - n_actions] + [1] * n_actions, dim=2)
        actions = Actions(*actions)

        # parse non-action inputs
        inputs = Obs(*torch.split(inputs, self.obs_sections, dim=2))
        obs = inputs.base.view(T, N, *self.obs_shape)
        task = inputs.task.view(T, N, self.n_subtasks,
                                self.obs_spaces.subtask.nvec.size)

        # build M
        task = torch.split(task, 1, dim=-1)
        interaction, count, obj = [x[0, :, :, 0] for x in task]
        M123 = torch.stack([interaction, count, obj], dim=-1)
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g123 = (interaction, count, obj)
        M = g123_to_binary(g123, one_hots)

        # build C
        branches = Branch(*torch.split(inputs.control, 1, dim=-1))
        conditions = self.condition_one_hots[branches.condition.long()]
        true_path = self.branch_one_hots[branches.true_path.long()]
        false_path = self.branch_one_hots[branches.false_path.long()]
        paths = torch.clamp(true_path + false_path, max=1)
        C = conditions.unsqueeze(1) * paths.unsqueeze(2)

        # parse hidden
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)
        p = hx.p
        r = hx.r
        for x in hx:
            x.squeeze_(0)
        if torch.any(new_episode):
            p[new_episode, 0] = 1.  # initialize pointer to first subtask
            r[new_episode] = M[new_episode, 0]  # initialize r to first subtask
            # initialize g to first subtask
            hx.g[new_episode] = 0.

        def update_attention(p, t):
            o = obs[t].unsqueeze(1)
            c = C.unsqueeze(0  # N
                            ).unsqueeze(3).unsqueeze(4)  # h, w
            logits = self.phi_shift(o * c)
            resolutions = torch.softmax(logits, dim=1)
            return p @ resolutions

        return self.pack(
            self.inner_loop(
                a=hx.a,
                g=hx.g,
                M=M,
                M123=M123,
                N=N,
                T=T,
                float_subtask=hx.subtask,
                next_subtask=inputs.next_subtask,
                obs=obs,
                p=p,
                r=r,
                actions=actions,
                update_attention=update_attention))
