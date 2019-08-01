from collections import namedtuple

import numpy as np
import torch
from torch.nn import functional as F

from ppo.agent import Agent
from ppo.control_flow.wrappers import Actions
from ppo.utils import broadcast3d

Obs = namedtuple("Obs", "base subtask")


class LowerLevel(Agent):
    def __init__(self, obs_spaces: Obs, action_spaces, **kwargs):
        # noinspection PyProtectedMember
        self.subtask_nvec = obs_spaces.subtask.nvec
        self.obs_spaces = obs_spaces
        self.obs_sections = [int(np.prod(s.shape)) for s in self.obs_spaces]
        _, h, w = self.obs_shape = self.obs_spaces.base.shape
        super().__init__(
            obs_shape=(self.d, h, w), action_space=action_spaces.a, **kwargs
        )

        for i, d in enumerate(self.subtask_nvec):
            self.register_buffer(f"part{i}_one_hot", torch.eye(int(d)))

    @property
    def d(self):
        return self.obs_spaces.base.shape[0] + int(  # base observation channels
            self.subtask_nvec.sum()
        )  # one-hot subtask

    def preprocess_obs(self, inputs):
        if not isinstance(inputs, Obs):
            n = inputs.size(0)
            inputs = torch.split(inputs, self.obs_sections, dim=-1)
            inputs = {
                k: x.view(n, *s.shape)
                for (k, s), x in zip(self.obs_spaces._asdict().items(), inputs)
            }
            g123 = inputs["control_flow"][torch.arange(n), inputs["subtask"].long()]
            g123 = torch.split(g123, 1, dim=-1)
            g123 = [x.flatten() for x in g123]
            g_binary = g_discrete_to_binary(
                g123, [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
            )
            inputs = Obs(base=inputs["base"], subtask=g_binary)

        g_broad = broadcast3d(inputs.subtask, self.obs_shape[-2:])
        obs = inputs.base.view(inputs.base.size(0), *self.obs_shape)
        return torch.cat([obs, g_broad], dim=1)

    def forward(self, inputs, z, *args, action=None, **kwargs):
        if action is not None:
            action = action[:, :1]
        act = super().forward(
            self.preprocess_obs(inputs), action=action, *args, **kwargs
        )
        x = torch.zeros_like(act.action)
        actions = Actions(a=act.action, g=x, cg=x, cr=x, z=z, l=x)
        return act._replace(action=torch.cat(actions, dim=-1))

    def get_value(self, inputs, rnn_hxs, masks):
        return super().get_value(self.preprocess_obs(inputs), rnn_hxs, masks)


def g_binary_to_discrete(g_binary, subtask_space):
    g123 = g_binary.nonzero()[:, 1:].view(-1, 3)
    g123 -= F.pad(torch.cumsum(subtask_space, dim=0)[:2], [1, 0])
    return g123


def g_discrete_to_binary(g_discrete, one_hots):
    return torch.cat(
        [one_hot[g.long()] for one_hot, g in zip(one_hots, g_discrete)], dim=-1
    )
