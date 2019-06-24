import numpy as np
import torch
from torch.nn import functional as F

from ppo.agent import Agent
from ppo.utils import broadcast3d
from ppo.wrappers import SubtasksActions, SubtasksObs


class SubtasksTeacher(Agent):
    def __init__(self, obs_space, action_space, **kwargs):
        self.obs_spaces = SubtasksObs(*obs_space.spaces)
        d, h, w = self.obs_shape = self.obs_spaces.base.shape
        self.obs_sections = SubtasksObs(
            base=d * h * w,
            subtask=int(self.obs_spaces.subtask.nvec.sum()),
            task=int(np.prod(self.obs_spaces.task.nvec.shape)),
            next_subtask=1,
        )
        self.action_spaces = SubtasksActions(*action_space.spaces)
        super().__init__(
            obs_shape=(self.d, h, w),
            action_space=self.action_spaces.a,
            **kwargs)

        for i, d in enumerate(self.obs_spaces.subtask.nvec):
            self.register_buffer(f'part{i}_one_hot', torch.eye(int(d)))

    @property
    def d(self):
        return self.obs_spaces.base.shape[0] + self.obs_sections.subtask

    def preprocess_obs(self, inputs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        obs = obs.view(-1, *self.obs_shape)
        subtasks = broadcast3d(subtasks, (self.obs_shape[-2:]))
        return torch.cat([obs, subtasks], dim=1)

    def forward(self, inputs, *args, action=None, **kwargs):
        if action is not None:
            action = action[:, :1]

        act = super().forward(
            self.preprocess_obs(inputs), action=action, *args, **kwargs)
        x = torch.zeros_like(act.action)
        actions = SubtasksActions(a=act.action, g=x, cg=x, cr=x)
        return act._replace(action=torch.cat(actions, dim=-1))

    def get_value(self, inputs, rnn_hxs, masks):
        return super().get_value(self.preprocess_obs(inputs), rnn_hxs, masks)


def g_binary_to_123(g_binary, subtask_space):
    g123 = g_binary.nonzero()[:, 1:].view(-1, 3)
    g123 -= F.pad(
        torch.cumsum(subtask_space, dim=0)[:2], [1, 0], 'constant', 0)
    return g123
