import numpy as np
import torch
import torch.nn as nn

from gridworld_env.control_flow_gridworld import Obs
from ppo.layers import Flatten, Parallel, Product, Reshape, ShallowCopy, Sum
import ppo.subtasks.agent
from ppo.utils import init_


class Agent(ppo.subtasks.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.subtasks.agent.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.obs_sections = [int(np.prod(s.shape)) for s in self.obs_spaces]
        self.register_buffer("branch_one_hots", torch.eye(self.n_subtasks))
        self.register_buffer("condition_one_hots", torch.eye(self.condition_size))
        self.register_buffer(
            "rows", torch.arange(self.n_subtasks).unsqueeze(-1).float()
        )

        d, h, w = self.obs_shape
        self.phi_shift = nn.Sequential(
            init_(nn.Linear(1, 1), "sigmoid"),
            # Parallel(
            # nn.Sequential(Reshape(1, d, h, w)),
            # nn.Sequential(Reshape(self.condition_size, 1, 1, 1)),
            # ),
            # Product(),
            # Reshape(d * self.condition_size, *self.obs_shape[-2:]),
            # init_(
            # nn.Conv2d(self.condition_size * d, hidden_size, kernel_size=1, stride=1)
            # ),
            # # attention {
            # ShallowCopy(2),
            # Parallel(
            # Reshape(hidden_size, h * w),
            # nn.Sequential(
            # init_(nn.Conv2d(hidden_size, 1, kernel_size=1)),
            # Reshape(1, h * w),
            # nn.Softmax(dim=-1),
            # ),
            # ),
            # Product(),
            # Sum(dim=-1),
            # # }
            # nn.ReLU(),
            # Flatten(),
            # init_(nn.Linear(hidden_size, 1), "sigmoid"),
            nn.Sigmoid(),
            Reshape(1, 1),
        )

    @property
    def condition_size(self):
        return int(self.obs_spaces.subtasks.nvec[0, 2])

    def parse_inputs(self, inputs):
        return Obs(*torch.split(inputs, self.obs_sections, dim=2))

    def inner_loop(self, inputs, **kwargs):
        N = inputs.base.size(1)

        # build C
        conditions = self.condition_one_hots[inputs.conditions[0].long()]
        control = inputs.control[0].view(N, *self.obs_spaces.control.nvec.shape)
        rows = self.rows.expand_as(control)
        control = control.where(control < self.n_subtasks, rows)
        false_path, true_path = torch.split(control, 1, dim=-1)
        true_path = self.branch_one_hots[true_path.squeeze(-1).long()]
        false_path = self.branch_one_hots[false_path.squeeze(-1).long()]

        def update_attention(p, t):
            c = (p.unsqueeze(1) @ conditions).squeeze(1)
            # phi_in = inputs.base[t, :, 1:-2] * c.view(N, conditions.size(2), 1, 1)
            # truth = torch.any(phi_in.view(N, -1) > 0, dim=-1).float().view(N, 1, 1)
            pred = self.phi_shift((inputs.base[t], c))
            trans = pred * true_path + (1 - pred) * false_path
            return (p.unsqueeze(1) @ trans).squeeze(1)

        kwargs.update(update_attention=update_attention)
        yield from super().inner_loop(inputs=inputs, **kwargs)
