import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld_env.control_flow_gridworld import Obs
from ppo.layers import Parallel, Product, Reshape
import ppo.subtasks.agent
from ppo.utils import init_


class Agent(ppo.subtasks.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.subtasks.agent.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.obs_sections = Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])
        self.register_buffer("branch_one_hots", torch.eye(self.n_subtasks))
        num_object_types = int(self.obs_spaces.subtasks.nvec[0, 2])
        self.register_buffer('condition_one_hots', torch.eye(num_object_types))
        self.register_buffer('rows', torch.arange(self.n_subtasks).unsqueeze(-1).float())
        self.n_conditions = self.obs_spaces.conditions.shape[0]

        d, h, w = self.obs_shape
        self.phi_shift = nn.Sequential(
            # Reshape(-1, num_object_types * d, h, w),
            Parallel(
                nn.Sequential(Reshape(-1, 1, d, h, w)),
                nn.Sequential(Reshape(-1, num_object_types, 1, 1, 1)),
            ),
            Product(),
            Reshape(-1, num_object_types * d * h * w),
            init_(nn.Linear(num_object_types * d * h * w, 1), "sigmoid"),
            # Reshape(-1, in_channels, *self.obs_shape[-2:]),
            # init_(
            # nn.Conv2d(num_object_types * d, hidden_size, kernel_size=1, stride=1)
            # ),
            # nn.MaxPool2d(kernel_size=self.obs_shape[-2:], stride=1),
            # Flatten(),
            # init_(nn.Linear(hidden_size, 1), "sigmoid"),
            nn.Sigmoid(),
            Reshape(-1, 1, 1),
        )
        self.obs_shapes = Obs(
            base=self.obs_spaces.base.shape,
            subtask=[1],
            subtasks=self.obs_spaces.subtasks.nvec.shape,
            conditions=self.obs_spaces.conditions.nvec.shape,
            control=self.obs_spaces.control.nvec.shape,
            next_subtask=[1],
            pred=[1],
        )

    # def get_obs_sections(self):
    # return Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])

    # def parse_inputs(self, inputs):
    # return Obs(*torch.split(inputs, self.obs_sections, dim=2))

    def inner_loop(self, inputs, **kwargs):
        N = inputs.base.size(1)

        # build C
        conditions = self.condition_one_hots[inputs.conditions[0].long()]
        control = inputs.control[0].view(N, *self.obs_spaces.control.nvec.shape)
        rows = self.rows.expand_as(control)
        # point terminal branches back at themselves TODO: is this right?
        control = control.where(control < self.n_subtasks, rows)
        false_path, true_path = torch.split(control, 1, dim=-1)
        true_path = self.branch_one_hots[true_path.squeeze(-1).long()]
        false_path = self.branch_one_hots[false_path.squeeze(-1).long()]

        def update_attention(p, t):
            c = (p.unsqueeze(1) @ conditions).squeeze(1)
            phi_in = (
                inputs.base[t, :, 1:-2] * c.view(N, conditions.size(2), 1, 1)
            ).view(N, -1)
            truth = torch.any(phi_in > 0, dim=-1).float().view(N, 1, 1)
            # pred = self.phi_shift((inputs.base[t], c))
            pred = truth
            trans = pred * true_path + (1 - pred) * false_path
            return (p.unsqueeze(1) @ trans).squeeze(1)

        return self.pack(
            self.inner_loop(
                new_episode=new_episode.unsqueeze(1).float(),
                a=hx.a,
                cr=hx.cr,
                cg=hx.cg,
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
                update_attention=update_attention,
            )
        )
