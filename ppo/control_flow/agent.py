import numpy as np
import torch
import torch.nn as nn

from gridworld_env.control_flow_gridworld import Branch
from ppo.control_flow.wrappers import Obs
from ppo.layers import Flatten, Reshape
import ppo.subtasks.agent
from ppo.subtasks.teacher import g123_to_binary
from ppo.subtasks.wrappers import Actions
from ppo.utils import init_


class Agent(ppo.subtasks.agent.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.subtasks.agent.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.obs_sections = Obs(
            *[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.register_buffer('branch_one_hots', torch.eye(self.n_subtasks))
        num_object_types = int(self.obs_spaces.subtask.nvec[2])
        self.register_buffer(
            'condition_one_hots',
            torch.eye(num_object_types + 1))  # +1 for determinism

        in_channels = (
            self.obs_shape[0] *  # observation
            (num_object_types + 1))  # condition tensor d
        self.phi_shift = nn.Sequential(
            Reshape(-1, in_channels, *self.obs_shape[-2:]),
            init_(
                nn.Conv2d(in_channels, hidden_size, kernel_size=1, stride=1)),
            nn.MaxPool2d(kernel_size=self.obs_shape[-2:], stride=1),
            Flatten(),
            init_(nn.Linear(hidden_size, 1), 'sigmoid'),
            nn.Sigmoid(),
            Reshape(-1, 1, 1),
        )
        self.n_conditions = self.obs_spaces.control.shape[0]

    @property
    def n_subtasks(self):
        return self.obs_spaces.subtasks.nvec.shape[0]

    def get_obs_sections(self):
        return Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])

    def register_agent_dummy_values(self):
        self.register_buffer(
            'agent_dummy_values',
            torch.zeros(
                1,
                sum([
                    self.obs_sections.subtasks,
                    self.obs_sections.control,
                    self.obs_sections.next_subtask,
                ])))

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
        subtasks = inputs.subtasks.view(T, N, self.n_subtasks,
                                        self.obs_spaces.subtask.nvec.size)

        # build M
        subtasks = torch.split(subtasks, 1, dim=-1)
        interaction, count, obj = [x[0, :, :, 0] for x in subtasks]
        M123 = torch.stack([interaction, count, obj], dim=-1)
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g123 = (interaction, count, obj)
        M = g123_to_binary(g123, one_hots)

        # build C
        control = inputs.control[0].view(N, self.n_conditions, 3)
        # deterministic = torch.tensor([[[
        #     0.,
        #     self.n_subtasks,
        #     self.n_subtasks,
        # ]]]).expand(N, self.n_subtasks - 1, 3)
        # control = torch.cat([control, deterministic], dim=1)
        branches = Branch(*torch.split(control, 1, dim=-1))
        for x in branches:
            x.squeeze_(2)
        conditions = self.condition_one_hots[branches.condition.long()]
        true_path = self.branch_one_hots[branches.true_path.long()]
        false_path = self.branch_one_hots[branches.false_path.long()]

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
            o = obs[t].unsqueeze(2)
            initial, rest = torch.split(
                conditions, [1, self.n_conditions - 1], dim=1)
            c = (new_episode.unsqueeze(-1).unsqueeze(-1).float() *
                 initial).unsqueeze(-1).unsqueeze(-1)
            # TODO: handle subsequent conditions
            # * (p[p + 1 <= rest.size(1)].unsqueeze(1)
            #                     @ rest).unsqueeze(-1).unsqueeze(-1)
            pred = self.phi_shift(o * c)
            trans = pred * true_path + (1 - pred) * false_path
            return (trans @ p.unsqueeze(-1)).squeeze(1)

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
