import numpy as np
import torch
from torch.nn import functional as F

from ppo.agent import Agent
from ppo.subtasks.wrappers import Actions
from ppo.utils import broadcast3d


class Teacher(Agent):
    def __init__(self, obs_spaces, action_space, **kwargs):
        self.obs_spaces = obs_spaces
        _, h, w = self.obs_shape = self.obs_spaces.base.shape
        self.action_spaces = Actions(*action_space.spaces)
        self.obs_sections = [int(np.prod(s.shape)) for s in self.obs_spaces]
        self.subtask_nvec = obs_spaces.task.nvec[0]
        super().__init__(obs_shape=(self.d, h, w),
                         action_space=self.action_spaces.a,
                         **kwargs)

        for i, d in enumerate(self.subtask_nvec):
            self.register_buffer(f'part{i}_one_hot', torch.eye(int(d)))

    @property
    def d(self):
        return (self.obs_spaces.base.shape[0] +  # base observation channels
                int(self.subtask_nvec.sum()))  # one-hot subtask

    def preprocess_obs(self, inputs):
        obs, subtask_idx, subtasks = torch.split(inputs,
                                                 self.obs_sections,
                                                 dim=1)[:3]
        n = obs.size(0)
        obs = obs.view(n, *self.obs_shape)
        subtask_idx = subtask_idx.long().flatten()
        subtasks = subtasks.view(n, *self.obs_spaces.task.shape)
        g123 = subtasks[torch.arange(n), subtask_idx]
        g123 = [x.flatten() for x in torch.split(g123, 1, dim=-1)]
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g_binary = g123_to_binary(g123, one_hots)
        g_broad = broadcast3d(g_binary, self.obs_shape[-2:])
        return torch.cat([obs, g_broad], dim=1)

    def forward(self, inputs, *args, action=None, **kwargs):
        if action is not None:
            action = action[:, :1]
        act = super().forward(self.preprocess_obs(inputs),
                              action=action,
                              *args,
                              **kwargs)
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
    return torch.cat([one_hot[g.long()] for one_hot, g in zip(one_hots, g123)],
                     dim=-1)
