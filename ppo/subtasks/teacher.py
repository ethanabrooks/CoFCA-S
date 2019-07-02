from collections import namedtuple

import numpy as np
import torch
from torch.nn import functional as F

from ppo.agent import Agent
from ppo.subtasks.wrappers import Actions
from ppo.utils import broadcast3d

Obs = namedtuple("Obs", "base subtask subtasks")


class Teacher(Agent):
    def __init__(self, obs_spaces, action_space, **kwargs):
        # noinspection PyProtectedMember
        self.original_obs_spaces = obs_spaces
        self.obs_spaces = Obs(
            base=obs_spaces.base,
            subtask=obs_spaces.subtask,
            subtasks=obs_spaces.subtasks,
        )
        _, h, w = self.obs_shape = self.obs_spaces.base.shape
        self.action_spaces = Actions(**action_space.spaces)

        def obs_sections(spaces):
            for s in spaces:
                yield int(np.prod(s.shape))

        self.original_obs_sections = list(obs_sections(obs_spaces))
        self.obs_sections = list(obs_sections(self.obs_spaces))
        self.subtask_dim = 3
        self.subtask_nvec = obs_spaces.subtasks.nvec[0, : self.subtask_dim]
        super().__init__(
            obs_shape=(self.d, h, w), action_space=self.action_spaces.a, **kwargs
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
            # training teacher (not running with metacontroller)
            inputs = list(torch.split(inputs, self.original_obs_sections, dim=-1))
            fields_ = {
                k: x.view(-1, *s.shape)
                for x, (k, s) in zip(inputs, self.original_obs_spaces._asdict().items())
                if k in self.obs_spaces._fields
            }
            inputs = Obs(**fields_)

        n = inputs.base.size(0)
        obs = inputs.base.view(n, *self.obs_shape)
        subtask_idx = inputs.subtask.long().flatten()
        subtasks = inputs.subtasks.view(n, *self.obs_spaces.subtasks.shape)[
            :, : self.subtask_dim
        ]
        g123 = subtasks[torch.arange(n), subtask_idx]
        g123 = [x.flatten() for x in torch.split(g123, 1, dim=-1)]
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g_binary = g123_to_binary(g123, one_hots)
        g_broad = broadcast3d(g_binary, self.obs_shape[-2:])
        return torch.cat([obs, g_broad], dim=1)

    def forward(self, inputs, *args, action=None, **kwargs):
        sections = self.obs_sections + [inputs.size(1) - sum(self.obs_sections)]
        inputs = Obs(*torch.split(inputs, sections, dim=1)[:-1])
        if action is not None:
            action = action[:, :1]
        act = super().forward(
            self.preprocess_obs(inputs), action=action, *args, **kwargs
        )
        x = torch.zeros_like(act.action)
        actions = Actions(a=act.action, g=x, cg=x, cr=x)
        return act._replace(action=torch.cat(actions, dim=-1))

    def get_value(self, inputs, rnn_hxs, masks):
        return super().get_value(self.preprocess_obs(inputs), rnn_hxs, masks)


def g_binary_to_123(g_binary, subtask_space):
    g123 = g_binary.nonzero()[:, 1:].view(-1, 3)
    g123 -= F.pad(torch.cumsum(subtask_space, dim=0)[:2], [1, 0])
    return g123


def g123_to_binary(g123, one_hots):
    return torch.cat([one_hot[g.long()] for one_hot, g in zip(one_hots, g123)], dim=-1)
