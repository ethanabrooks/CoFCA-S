import numpy as np
import torch
import torch.nn as nn

from ppo.control_flow.wrappers import Obs
from ppo.layers import Flatten, Reshape
import ppo.subtasks.agent
from ppo.subtasks.teacher import g123_to_binary
from ppo.subtasks.wrappers import Actions
from ppo.utils import init_


class Agent(ppo.subtasks.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.subtasks.agent.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.register_buffer('branch_one_hots', torch.eye(self.n_subtasks))
        num_object_types = int(self.obs_spaces.subtasks.nvec[0, 2])
        self.register_buffer('condition_one_hots',
                             torch.eye(num_object_types + 1))  # +1 for determinism
        self.register_buffer('rows', torch.arange(self.n_subtasks).unsqueeze(-1).float())

        in_channels = (
            self.obs_shape[0] *  # observation
            (num_object_types + 1))  # condition tensor d
        self.phi_shift = nn.Sequential(
            Reshape(-1, in_channels, *self.obs_shape[-2:]),
            init_(nn.Conv2d(in_channels, hidden_size, kernel_size=1, stride=1)),
            nn.MaxPool2d(kernel_size=self.obs_shape[-2:], stride=1),
            Flatten(),
            init_(nn.Linear(hidden_size, 1), 'sigmoid'),
            nn.Sigmoid(),
            Reshape(-1, 1, 1),
        )
        self.n_conditions = self.obs_spaces.conditions.shape[0]
        self.obs_shapes = Obs(
            base=self.obs_spaces.base.shape,
            subtask=[1],
            subtasks=self.obs_spaces.subtasks.nvec.shape,
            conditions=self.obs_spaces.conditions.nvec.shape,
            control=self.obs_spaces.control.nvec.shape,
            next_subtask=[1],
            pred=[1],
        )

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
        inputs, *actions = torch.split(inputs.detach(), [D - n_actions] + [1] * n_actions, dim=2)
        actions = Actions(*actions)

        # parse non-action inputs
        inputs = torch.split(inputs, self.obs_sections, dim=2)
        inputs = Obs(*[x.view(T, N, *shape) for x, shape in zip(inputs, self.obs_shapes)])

        # build M
        subtasks = torch.split(inputs.subtasks, 1, dim=-1)
        interaction, count, obj = [x[0, :, :, 0] for x in subtasks]
        M123 = torch.stack([interaction, count, obj], dim=-1)
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g123 = (interaction, count, obj)
        M = g123_to_binary(g123, one_hots)

        # build C
        conditions = self.condition_one_hots[inputs.conditions[0].long()]
        control = inputs.control[0]
        rows = self.rows.expand_as(control)
        # point terminal branches back at themselves TODO: is this right?
        control = inputs.control[0].where(control != self.n_subtasks, rows)
        false_path, true_path = torch.split(control, 1, dim=-1)
        true_path = self.branch_one_hots[true_path.squeeze(-1).long()]
        false_path = self.branch_one_hots[false_path.squeeze(-1).long()]

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
            o = inputs.base[t].unsqueeze(2)
            c = (p.unsqueeze(1) @ conditions).view(N, 1, -1, 1, 1)
            pred = self.phi_shift(o * c)  # TODO
            # pred = inputs.pred[t].view(N, 1, 1)
            trans = pred * true_path + (1 - pred) * false_path
            return (p.unsqueeze(1) @ trans).squeeze(1)

        return self.pack(
            self.inner_loop(a=hx.a,
                            g=hx.g,
                            M=M,
                            M123=M123,
                            N=N,
                            T=T,
                            float_subtask=hx.subtask,
                            next_subtask=inputs.next_subtask,
                            obs=inputs.base,
                            p=p,
                            r=r,
                            actions=actions,
                            update_attention=update_attention))
